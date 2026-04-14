import torch
import gradio as gr
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.integrated_system import MultimodalSystem

# Inicializamos el sistema (esto carga los 3 modelos)
try:
    system = MultimodalSystem()
except Exception as e:
    print(f"Error cargando el sistema: {e}")
    # Si falla por los archivos .pth, el sistema no arrancará
    system = None


def predict_image(img):
    """Función para la CNN: Recibe imagen, devuelve clase"""
    if system is None:
        return "Error: Modelos no cargados"

    # Preprocesar imagen para la CNN (32x32)
    img_input = img.resize((32, 32))
    img_tensor = torch.from_numpy(
        np.array(img_input)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(system.device)

    with torch.no_grad():
        output = system.cnn(img_tensor)
        probs = F.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return f"{system.categories[idx.item()]} ({conf.item()*100:.2f}%)"


def generate_full(category):
    """Función para GAN + LSTM: Genera imagen y texto"""
    if system is None:
        return None, "Error: Modelos no cargados"
    desc, img, conf = system.generate_and_verify(category)
    return img, f"{desc}\n\n[Validación CNN: {conf:.2f}% de parecido]"


# --- CSS Personalizado para Gradio ---
# Este CSS busca el contenedor de la imagen de salida de la GAN
# y fuerza al navegador a renderizarla grande (512px) y nítida.
css = """
#gan_output_img img {
    width: 512px !important;
    height: 512px !important;
    image-rendering: pixelated; /* Para Chrome/Edge */
    image-rendering: crisp-edges; /* Para Firefox */
    object-fit: contain;
}
"""

# --- Diseño de la Interfaz con Bloques y CSS ---
with gr.Blocks(title="IA Multimodal CIFAR-10", css=css) as demo:
    gr.Markdown("# 🧠 Sistema Integrado CNN-LSTM-GAN")

    with gr.Tab("🚀 Generador (GAN + LSTM)"):
        with gr.Row():
            with gr.Column():
                input_cat = gr.Dropdown(
                    ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck'],
                    label="Selecciona Categoría"
                )
                btn_gen = gr.Button("Generar Arte y Descripción")
            with gr.Column():
                # Le asignamos un ID específico 'gan_output_img' para el CSS
                out_img = gr.Image(
                    label="Imagen Creada por GAN (32x32 nativa)", elem_id="gan_output_img")
                out_text = gr.Textbox(label="Relato de la LSTM")

        btn_gen.click(generate_full, inputs=input_cat,
                      outputs=[out_img, out_text])

    with gr.Tab("🔍 Clasificador (CNN)"):
        with gr.Row():
            with gr.Column():
                input_file = gr.Image(
                    type="pil", label="Sube una imagen de 32x32")
                btn_class = gr.Button("Clasificar con VGG-CNN")
            with gr.Column():
                out_label = gr.Label(label="Predicción del Modelo")

        btn_class.click(predict_image, inputs=input_file, outputs=out_label)

if __name__ == "__main__":
    demo.launch()
