import csv
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import pywinstyles

# Faster R-CNN inference function
def run_fasterrcnn_inference(test_images_path, results_path='./results', confidence=0.50, model_path="fasterrcnn_resnet50_epoch_6.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=5)  # 4 classes + background
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    os.makedirs(os.path.join(results_path, 'test_predictions', 'labels'), exist_ok=True)

    class_names = ['Background', 'Switch', 'Crossing', 'Normal Signal', 'B2B Signal']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow

    for img_path in glob.glob(os.path.join(test_images_path, '*')):
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']

        # Filter predictions based on confidence threshold
        mask = scores > confidence
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Save results
        result = {
            'path': img_path,
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }
        results.append(result)

        # Save labeled image
        img_labeled = img.copy()
        draw = ImageDraw.Draw(img_labeled)
        font = ImageFont.load_default()

        for box, label, score in zip(boxes, labels, scores):
            box = box.cpu().numpy()
            label = label.item()
            score = score.item()
            
            # Draw bounding box and label on the image
            color = colors[label - 1]  # -1 because label 0 is background
            draw.rectangle(box.tolist(), outline=color, width=3)
            label_text = f"{class_names[label]}: {score:.2f}"
            draw.text((box[0], box[1] - 10), label_text, fill=color, font=font)

        img_labeled.save(os.path.join(results_path, 'test_predictions', os.path.basename(img_path)))

        # Save labels in YOLO format
        with open(os.path.join(results_path, 'test_predictions', 'labels', os.path.splitext(os.path.basename(img_path))[0] + '.txt'), 'w') as f:
            for box, label in zip(boxes, labels):
                x, y, w, h = box.tolist()
                f.write(f"{label.item() - 1} {(x + w/2)/img.width} {(y + h/2)/img.height} {w/img.width} {h/img.height}\n")

    return results

def summarize_results(results):
    counts = {'Switch': 0, 'Crossing': 0, 'Normal Signal': 0, 'B2B Signal': 0}
    for result in results:
        for label in result['labels']:
            class_name = ['Background', 'Switch', 'Crossing', 'Normal Signal', 'B2B Signal'][label.item()]
            if class_name in counts:
                counts[class_name] += 1
    return counts

def results_to_csv(results, latest_file):
    # Define CSV file path
    csv_file = os.path.join(latest_file, 'detection_results.csv')

    # Prepare CSV Header
    csv_data = [["Filename", "Switch", "Crossing", "Normal Signal", "B2B Signal"]]

    # Iterate through results list
    for result in results:
        file_path = result['path']
        filename = os.path.basename(file_path)

        # Extract detections
        switch = sum(1 for label in result['labels'] if label.item() == 1)
        crossing = sum(1 for label in result['labels'] if label.item() == 2)
        normal_signal = sum(1 for label in result['labels'] if label.item() == 3)
        b2b_signal = sum(1 for label in result['labels'] if label.item() == 4)

        # Append row to CSV data
        csv_data.append([filename, switch, crossing, normal_signal, b2b_signal])

    # Write to CSV file
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"CSV file saved as {csv_file}")

# GUI Class
class RailwaysAssetDetector(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Railways Asset Detector")
        self.geometry("950x650")

        self.set_background_image('./images/background.png')

        self.test_images_path = ''

        title = ctk.CTkLabel(self, text="Railways Asset Detector", font=('Arial', 34, 'bold'),bg_color="#F1F5FB", fg_color='transparent', text_color="black")
        title.pack(pady=30)
        pywinstyles.set_opacity(title, value=0.5, color="#000001")
        self.upload_btn = ctk.CTkButton(self, text="Upload Images Folder", font=('Arial', 16, 'bold'), command=self.upload_folder, bg_color="#F1F5FB", fg_color='#3B82F6')
        self.upload_btn.pack(pady=10)

        self.process_btn = ctk.CTkButton(self, text="Process", font=('Arial', 16, 'bold'), command=self.process_images, bg_color="#F1F5FB", fg_color='#FF9800', state='disabled')
        self.process_btn.pack(pady=10)

        self.progress_label = ctk.CTkLabel(self, text="", font=("Arial", 16), text_color="white")
        self.progress_label.pack(pady=10)

        # Cards Frame
        self.cards_frame = ctk.CTkFrame(self, bg_color='transparent', fg_color='transparent', corner_radius=20)
        self.cards_frame.pack(expand=True, pady=20)

        # Create result cards with images
        self.switch_label = self.create_card('Switches', './images/switch.png')
        self.crossing_label = self.create_card('Crossings', './images/crossing.png')
        self.normal_label = self.create_card('Normal Signals', './images/normalsignal.png')
        self.b2b_label = self.create_card('B2B Signals', './images/b2bsignal.png')

    def create_card(self, title, image_file):
        card_frame = ctk.CTkFrame(self.cards_frame, bg_color='transparent', fg_color='#F1F5FB', corner_radius=20, width=200, height=250)
        card_frame.pack(side='left', padx=15, pady=10)
        card_frame.pack_propagate(False)

        if os.path.exists(image_file):
            img = ctk.CTkImage(Image.open(image_file),  size=(100, 100))
            img_label = ctk.CTkLabel(card_frame, image=img, text='', fg_color='transparent', corner_radius=20)
            img_label.image = img  # Prevent garbage collection
            img_label.pack(pady=1)

        label_title = ctk.CTkLabel(card_frame, text=title, font=('Arial', 18, 'bold'), fg_color='transparent', text_color='black')
        label_title.pack(pady=5)

        label_count = ctk.CTkLabel(card_frame, text='0', font=('Arial', 30, 'bold'), text_color='black', fg_color='transparent')
        label_count.pack(pady=5)

        return label_count

    def upload_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.test_images_path = folder
            self.process_btn.configure(state='normal')

    def process_images(self):
        self.progress_label.configure(text="Processing, please wait...")
        self.update_idletasks()

        results = run_fasterrcnn_inference(self.test_images_path)
        counts = summarize_results(results)
        print(f"counts:{counts}")

        self.switch_label.configure(text=str(counts['Switch']))
        self.crossing_label.configure(text=str(counts['Crossing']))
        self.normal_label.configure(text=str(counts['Normal Signal']))
        self.b2b_label.configure(text=str(counts['B2B Signal']))

        list_of_files = glob.glob('./results/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        results_to_csv(results, latest_file)
        annotated_path = os.path.abspath(f'{latest_file}')
        self.progress_label.configure(text="")
        messagebox.showinfo("Process Completed", f"Annotated images saved at:\n{annotated_path}")

    def set_background_image(self, image_path):
        if os.path.exists(image_path):
            bg_image = Image.open(image_path)
            bg_photo = ctk.CTkImage(light_image=bg_image, size=(950, 650))
            bg_label = ctk.CTkLabel(self, image=bg_photo, text="")
            bg_label.place(relx=0.5, rely=0.5, anchor="center")
            bg_label.lower()

if __name__ == '__main__':
    app = RailwaysAssetDetector()
    app.mainloop()