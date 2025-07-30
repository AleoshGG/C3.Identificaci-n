import RPi.GPIO as GPIO
from ultralytics import YOLO
import cv2
import threading
import time
from typing import List

# Definición de pines
TRIG_PIN = 23
ECHO_PIN = 24
LED_PIN = 17

class Identification:
    def __init__(self):
        # Configuración de GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.setup(LED_PIN, GPIO.OUT)

        # Inicializar trigger en bajo
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.5)

        # Carga del modelo YOLO
        self.model = YOLO("yolo11n.pt")
        self.model.overrides["show"] = False

        self.running = True
        self.current_distance = 0.0
        self.sensor_lock = threading.Lock()

    def readSensor(self):
        """
        Lee el sensor ultrasónico HC‑SR04 y actualiza self.current_distance.
        Usa un protocolo de temporización más fiable y no reinicializa mal las marcas de tiempo.
        """
        while self.running:
            try:
                # 1) Asegurar TRIG en bajo antes de empezar
                GPIO.output(TRIG_PIN, False)
                time.sleep(0.05)

                # 2) Generar pulso de 10µs en TRIG
                GPIO.output(TRIG_PIN, True)
                time.sleep(0.00001)
                GPIO.output(TRIG_PIN, False)

                # 3) Esperar a que el ECHO cambie a alto (inicio del pulso)
                start_timeout = time.time() + 0.02
                while GPIO.input(ECHO_PIN) == 0 and time.time() < start_timeout:
                    pass
                t_start = time.time()

                # 4) Esperar a que el ECHO vuelva a bajo (fin del pulso)
                end_timeout = time.time() + 0.02
                while GPIO.input(ECHO_PIN) == 1 and time.time() < end_timeout:
                    pass
                t_end = time.time()

                # 5) Si alguno de los bucles hizo timeout, ignoramos esta lectura
                if t_end <= t_start:
                    print("Timeout en lectura de ECHO")
                else:
                    # 6) Calcular distancia (velocidad del sonido aprox. 34300 cm/s)
                    pulse_duration = t_end - t_start
                    distance = (pulse_duration * 34300) / 2  # medio recorrido
                    distance = round(distance, 2)

                    # 7) Comprobar rango válido
                    if 2 <= distance <= 400:
                        with self.sensor_lock:
                            self.current_distance = distance
                        print(f"Sensor leyendo: {distance} cm")
                    else:
                        print(f"Lectura fuera de rango: {distance} cm")

                # 8) Pausa antes de la siguiente medida
                time.sleep(0.2)

            except Exception as e:
                print(f"Error en sensor: {e}")
                time.sleep(0.5)

    def get_current_distance(self) -> float:
        with self.sensor_lock:
            return self.current_distance

    def turnOnLed(self):
        GPIO.output(LED_PIN, True)
        print("LED ENCENDIDO")
    
    def turnOffLed(self):
        GPIO.output(LED_PIN, False)
        print("LED APAGADO")
    
    def process_frame(self, frame) -> (any, List[int]):
        try:
            results = list(self.model.track(frame, stream=True, show=False))
            if not results:
                return frame, []

            det = results[0]
            cls_ids = det.boxes.cls.cpu().numpy().astype(int).tolist()
            annotated = det.plot()
            return annotated, cls_ids

        except Exception as e:
            print(f"Error procesando frame: {e}")
            return frame, []

    def start(self):
        # Hilo de sensor ultrasónico
        sensor_thread = threading.Thread(target=self.readSensor, daemon=True)
        sensor_thread.start()

        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Detección de personas", cv2.WINDOW_NORMAL)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while cam.isOpened() and self.running:
                ret, frame = cam.read()
                if not ret:
                    break

                annotated, detected_classes = self.process_frame(frame)
                distance = self.get_current_distance()

                # Verificar persona (clase 0 en COCO)
                person_detected = 0 in detected_classes
                print(f"Distacnia: {distance}")
                if person_detected and distance <= 100:
                    self.turnOnLed()
                else:
                    self.turnOffLed()

                cv2.imshow("Detección de personas", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("Programa interrumpido por el usuario")
        except Exception as e:
            print(f"Error en bucle principal: {e}")
        finally:
            self.cleanup(cam)

    def cleanup(self, cam):
        """Limpieza de recursos: cámara, ventanas y GPIOs"""
        self.running = False
        cam.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Recursos liberados correctamente")

if __name__ == "__main__":
    Identification().start()
