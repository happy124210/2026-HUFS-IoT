import RPi.GPIO as GPIO
import time
import threading

BUZZER_PIN = 11
LED_PIN = 13

_alert_active = False
_alert_lock = threading.Lock()
_pwm = None
_initialized = False

def init_gpio():
    global _pwm, _initialized
    if _initialized:
        return
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)
    _pwm = GPIO.PWM(BUZZER_PIN, 1000)
    _initialized = True

def start_alert():
    global _alert_active
    init_gpio()
    with _alert_lock:
        _alert_active = True
    t = threading.Thread(target=_alert_loop, daemon=True)
    t.start()

def stop_alert():
    global _alert_active
    with _alert_lock:
        _alert_active = False
    try:
        _pwm.stop()
        GPIO.output(LED_PIN, GPIO.LOW)
    except:
        pass

def _alert_loop():
    while True:
        with _alert_lock:
            if not _alert_active:
                break
        try:
            GPIO.output(LED_PIN, GPIO.HIGH)
            _pwm.start(50)
            time.sleep(0.5)
            GPIO.output(LED_PIN, GPIO.LOW)
            _pwm.stop()
            time.sleep(0.3)
        except Exception as e:
            print(f"GPIO 루프 오류: {e}")
            break

def cleanup():
    global _initialized
    stop_alert()
    try:
        GPIO.cleanup()
        _initialized = False
    except:
        pass

if __name__ == '__main__':
    print("부저 + LED 테스트... (3초간)")
    start_alert()
    time.sleep(3)
    stop_alert()
    cleanup()
    print("완료!")
