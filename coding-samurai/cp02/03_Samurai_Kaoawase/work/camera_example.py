import gradio as gr
import datetime

def shot(image):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_info = f"撮影時刻: {timestamp}, サイズ: {image.size}, タイプ: {type(image)}"
    return image, image_info

with gr.Blocks() as demo:
    camera_image = gr.Image(type="pil", sources=["webcam"], streaming=True, width=768, height=512)
    button_shot = gr.Button("Shot")
    output_image = gr.Image(type="pil")
    output_text = gr.Textbox(label="画像情報")
    button_shot.click(shot, camera_image, [output_image, output_text])

if __name__ == "__main__":
    demo.launch()
