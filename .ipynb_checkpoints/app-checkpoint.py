import torch
import gradio as gr
from model_0_weights.pth import ModelV1  # импорт модели из твоего файла

# Загружаем модель
model = ModelV1()
model.load_state_dict(torch.load("model_0_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Функция предсказания (например, для одного изображения)
def predict(image):
    # тут нужна твоя предобработка (преобразование изображения в тензор)
    # пример для PIL Image в grayscale 28x28:
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(image).unsqueeze(0)  # добавляем batch dimension
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    return pred

# Интерфейс Gradio
iface = gr.Interface(fn=predict,
                     inputs=gr.Image(type="pil"),
                     outputs="label")

iface.launch()