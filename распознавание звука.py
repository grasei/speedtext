import os 
import threading
import queue
import time
import sys
import ctypes
import winsound
import ctypes.wintypes
import uuid
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import ollama
from faster_whisper import WhisperModel

# Импорт для уведомлений
try:
    from windows_toasts import InteractableWindowsToaster, Toast
except ImportError:
    print("Ошибка: Библиотека windows-toasts не найдена.")
    print("Установите её командой: pip install windows-toasts")
    sys.exit(1)

# --- ИНИЦИАЛИЗАЦИЯ ---
mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "GemmaWhisper_V3_Stable")
if ctypes.windll.kernel32.GetLastError() == 183: 
    print("Программа уже запущена")
    time.sleep(1)
    sys.exit(0)

print("Загрузка моделей...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)

# Состояния
is_recording = False
is_paused = False
audio_buffer = []
audio_queue = queue.Queue()
esc_presses = []

# Состояния для коррекции
auto_correct_mode = False
scroll_lock_presses = []
correction_queue = queue.Queue()

# Очередь для уведомлений
notification_queue = queue.Queue()

# Блокировка для безопасной работы с буфером обмена
# Гарантирует, что автокоррекция не перезапишет текст во время вставки распознанного
clipboard_lock = threading.Lock()

def play_sound(action):
    s = {"start": [(440, 100), (660, 100)], "stop": [(660, 100), (440, 100)], 
         "pause": [(300, 150)], "resume": [(800, 150)], "fix": [(1000, 80), (1200, 80)],
         "mode_on": [(600, 100), (800, 100), (1000, 100)], 
         "mode_off": [(1000, 100), (800, 100), (600, 100)],
         "copy": [(1200, 50), (1400, 50)]}
    for f, d in s.get(action, []): 
        winsound.Beep(f, d)

# --- ПОТОК УВЕДОМЛЕНИЙ ---

def notification_worker():
    """
    Владеет объектом toaster. Просто показывает уведомления.
    """
    ctypes.windll.ole32.CoInitializeEx(None, 2)
    toaster = InteractableWindowsToaster('GemmaWhisper')
    active_toasts = []
    msg = ctypes.wintypes.MSG()
    
    while True:
        # 1. Message Pump
        if ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        
        # 2. Обработка очереди
        try:
            # Получаем только заголовок и тело
            title_str, body_str = notification_queue.get_nowait()
            
            newToast = Toast()
            newToast.Tag = str(uuid.uuid4())
            newToast.text_fields = [title_str, body_str]
            
            # Кнопок нет, просто информация
            
            active_toasts.append(newToast)
            if len(active_toasts) > 20:
                active_toasts.pop(0)

            toaster.show_toast(newToast)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Ошибка внутри потока уведомлений: {e}")
        
        time.sleep(0.01)

threading.Thread(target=notification_worker, daemon=True).start()

def push_toast(title, body):
    notification_queue.put((title, body))

# --- ФУНКЦИИ В ОТДЕЛЬНЫХ ПОТОКАХ ---

def correction_daemon():
    while True:
        text = correction_queue.get()
        if not text:
            continue
            
        print(f"Корректор проверяет: {text}")
        try:
            res = ollama.chat(model='gemma2:2b', messages=[
                {'role': 'system', 'content': 'You are a Russian spellchecker. Correct grammar and punctuation. Return ONLY the corrected text.'},
                {'role': 'user', 'content': text}], options={'temperature': 0})
            fixed = res['message']['content'].strip()
            
            if fixed and fixed != text:
                print(f"Найдена правка: {fixed}")
                
                # БЛОКИРОВКА: Копируем в буфер безопасно
                with clipboard_lock:
                    pyperclip.copy(fixed)
                
                play_sound("copy")
                print("Текст скопирован в буфер (доступен через Win+V)")
                
                # Уведомление
                title_str = f"Правка: {(text[:20] + '..') if len(text) > 20 else text}"
                body_str = f"Скопировано в буфер:\n{fixed}\n\n(Нажмите Win+V для вставки)"
                push_toast(title_str, body_str)
            else:
                print("Текст совпадает, пропускаем.")
                
        except Exception as e:
            print(f"Ошибка коррекции: {e}")

threading.Thread(target=correction_daemon, daemon=True).start()

def async_toggle_recording():
    global is_recording, audio_buffer, is_paused
    if not is_recording:
        play_sound("start")
        print("Начало записи...")
        push_toast("Диктовка", "Начало записи...")
        is_recording, is_paused, audio_buffer = True, False, []
        threading.Thread(target=record_loop, daemon=True).start()
    else:
        is_recording = False
        play_sound("stop")
        print("Завершение записи...")
        push_toast("Диктовка", "Запись остановлена. Обработка...")
        threading.Thread(target=process_audio, daemon=True).start()

def record_loop():
    with sd.InputStream(samplerate=16000, channels=1, callback=lambda i,f,t,s: audio_queue.put(i.copy()) if is_recording and not is_paused else None):
        while is_recording:
            while not audio_queue.empty(): 
                audio_buffer.append(audio_queue.get())
            time.sleep(0.1)

def process_audio():
    global audio_buffer
    if not audio_buffer: 
        return
    data = np.concatenate(audio_buffer, axis=0).flatten()
    segments, _ = whisper_model.transcribe(data, beam_size=5, language="ru")
    text = " ".join([s.text for s in segments]).strip()
    
    if text:
        # БЛОКИРОВКА: Вставка происходит атомарно
        with clipboard_lock:
            pyperclip.copy(text)
            keyboard.press_and_release('ctrl+v')
        
        print(f"Распознано: {text}")
        
        if auto_correct_mode:
            print("Добавлено в очередь на коррекцию.")
            correction_queue.put(text)

# --- ОБРАБОТЧИК КЛАВИШ ---
def on_key_event(e):
    global is_paused, esc_presses, is_recording, auto_correct_mode, scroll_lock_presses
    
    if e.event_type == 'down':
        if e.name == 'scroll lock':
            now = time.time()
            if scroll_lock_presses and now - scroll_lock_presses[-1] > 0.5:
                scroll_lock_presses.clear()
            
            scroll_lock_presses.append(now)
            
            if len(scroll_lock_presses) == 2:
                scroll_lock_presses.clear()
                auto_correct_mode = not auto_correct_mode
                if auto_correct_mode:
                    play_sound("mode_on")
                    print("--- РЕЖИМ АВТОКОРРЕКЦИИ ВКЛЮЧЕН ---")
                    push_toast("Автокоррекция", "Режим ВКЛЮЧЕН")
                else:
                    play_sound("mode_off")
                    print("--- РЕЖИМ АВТОКОРРЕКЦИИ ВЫКЛЮЧЕН ---")
                    push_toast("Автокоррекция", "Режим ВЫКЛЮЧЕН")
            
        elif e.name == 'print screen': 
            async_toggle_recording()

        elif e.name == 'right ctrl' and is_recording: 
            is_paused = not is_paused
            play_sound("pause" if is_paused else "resume")
            print("Пауза") if is_paused else print("Продолжение")
            
        elif e.name == 'esc':
            now = time.time()
            if len(esc_presses) > 0 and now - esc_presses[-1] > 1.5:
                esc_presses.clear()
            esc_presses.append(now)
            if len(esc_presses) >= 3:
                winsound.Beep(200, 600)
                os._exit(0)

keyboard.hook(on_key_event)
print("Система готова.")
print("Print Screen    - Запись")
print("Right Ctrl      - Пауза")
print("Scroll Lock x2  - Вкл/Выкл автокоррекцию")
print("Esc три раза    - Выход")
keyboard.wait()
