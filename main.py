
from ultralytics import YOLO
import cv2
import threading
import time
import queue
from typing import Optional

class Identification:
    def __init__(self): 
        self.model = YOLO("yolo11n.pt")
        self.distance_queue = queue.Queue(maxsize=1)  # Cola para comunicación entre hilos
        self.running = True
        self.current_distance = 0.0
        self.sensor_lock = threading.Lock()
        
    def readSensor(self):
        # Hilo dedicado para lectura del sensor Ultrasónico
        while self.running:
            try:
               
                distance = 50.5  
                print(f"Sensor leyendo: {distance}")
                
                # Actualizar distancia de forma thread-safe
                with self.sensor_lock:
                    self.current_distance = distance
                    
                time.sleep(0.1)  # Reducido para mayor frecuencia de lectura
                
            except Exception as e:
                print(f"Error en sensor: {e}")
                time.sleep(0.5)

    def get_current_distance(self) -> float:
        # Obtener la distancia actual de forma thread-safe
        with self.sensor_lock:
            return self.current_distance

    def turnOnLed(self):
        # Encender el led
        print("LED ENCENDIDO")
    
    def turnOffLed(self):
        # Apagar el led
        print("LED APAGADO")
    
    def process_frame(self, frame):
        # Procesar frame de video - puede ejecutarse en hilo separado
        try:
            results = list(self.model.track(frame, stream=True))
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                annotated = results[0].plot()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                return annotated, ids
            else:
                return frame, []
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return frame, []
    
    def start(self):
        # Iniciar hilo del sensor
        sensor_thread = threading.Thread(target=self.readSensor, daemon=True)
        sensor_thread.start()
        
        # Pequeña pausa para que el sensor inicie
        time.sleep(0.5)
        
        cam = cv2.VideoCapture(0)
        
        # Optimizar configuración de cámara
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia
        cam.set(cv2.CAP_PROP_FPS, 30)  # Configurar FPS
        
        try:
            while cam.isOpened() and self.running:
                ret, frame = cam.read()
                if not ret: 
                    break

                # Procesar frame (detección YOLO)
                annotated, detected_ids = self.process_frame(frame)
                
                # Obtener distancia actual
                distance = self.get_current_distance()
                
                # Lógica de control del LED
                person_detected = 1 in detected_ids  

                if person_detected and distance <= 100:
                    self.turnOnLed()
                else:
                    self.turnOffLed()

                # Mostrar frame
                cv2.imshow("Detección de personas", annotated)

                # Salir con ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("Programa interrumpido por el usuario")
        except Exception as e:
            print(f"Error en bucle principal: {e}")
        finally:
            self.cleanup(cam)

    def cleanup(self, cam):
        """Limpieza de recursos"""
        self.running = False
        cam.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente")

if __name__ == "__main__":
   identification = Identification()
   identification.start()