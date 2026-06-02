import cv2
from ultralytics import YOLO

def iniciar_scanner():
    # 1. Carregamento do modelo YOLO (utilizando o modelo 'nano' por ser ideal para processamento leve)
    # Carrega os pesos pré-treinados da rede YOLOv8n
    print("Carregando o modelo YOLOv8...")
    model = YOLO("yolov8n.pt")

    # 2. Inicialização da câmera da máquina
    # Captura a câmera padrão do sistema (índice 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera do dispositivo.")
        return

    print("Câmera iniciada com sucesso. Pressione 'q' na janela de vídeo para encerrar.")

    # 3. Processamento do fluxo de vídeo e detecção de objetos
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler o frame da câmera.")
            break

        # Executa a inferência de detecção de objetos no frame atual
        resultados = model(frame, verbose=False)

        # Renderiza os boxes e labels dos objetos detectados diretamente no frame
        frame_anotado = resultados[0].plot()

        # Exibe o frame processado em uma janela nativa do sistema operacional
        cv2.imshow("Scanner com YOLOv8 - Pressione 'q' para sair", frame_anotado)

        # Interrompe o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Liberação dos recursos de hardware da câmera e fechamento das janelas
    cap.release()
    cv2.destroyAllWindows()
    print("Scanner encerrado com sucesso.")

if __name__ == "__main__":
    iniciar_scanner()