import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
from matplotlib.figure import Figure
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QMenuBar, QDialog, QLabel, QVBoxLayout,
    QMessageBox, QFileDialog, QWidget, QGridLayout, QInputDialog, QHBoxLayout, QPushButton, QDialogButtonBox, QCheckBox,
    QComboBox, QSlider, QTextEdit, QLineEdit
)
from PyQt5.QtGui import QFont, QPalette, QLinearGradient, QColor, QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
import sqlite3
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QMovie

import matplotlib.pyplot as plt
plt.ion()
import torch
from yolov5.utils.general import non_max_suppression
from yolov5.models.common import DetectMultiBackend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random



#### new class for histogram
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_histogram(self, normal_count, pneumonia_count):
        labels = ['Normal', 'Pneumonia']
        counts = [normal_count, pneumonia_count]

        self.axes.clear()
        bars = self.axes.bar(labels, counts, color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=1.2)

        for bar in bars:
            yval = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom')

        self.axes.set_xlabel('Lung Condition', fontsize=14, fontweight='bold')
        self.axes.set_ylabel('Count', fontsize=14, fontweight='bold')
        self.axes.set_title('Histogram of Lung Conditions', fontsize=16, fontweight='bold')
        self.axes.grid(True, which='both', linestyle='--', linewidth=0.5)

        self.draw()  # Update the canvas with the new plot



class ResultsDialog(QDialog):
    def __init__(self, parent=None, image_path=None, annotated_image_path=None, edge_image_path=None,
                 analysis_result=None):
        super().__init__(parent)
        self.setWindowTitle('Analysis Results')
        self.setStyleSheet("background-color: #f0f0f0; color: #333; font-family: Arial; font-size: 14px;")
        self.setGeometry(100, 100, 1200, 600)

        # Result label style update with gradient and icon
        self.setStyleSheet("""
                    QDialog {
                        background-color: #f5f5f5;
                        border-radius: 15px;
                        border: 2px solid #ccc;
                    }
                    QLabel {
                        font-size: 18px;
                        color: #333;
                        padding: 10px;
                        background-color: transparent;
                        border: none;
                    }
                    .ImageLabel {
                        background-color: transparent;
                        border-radius: 10px;
                        border: 2px solid #ccc;
                    }
                    .TextLabel {
                        font-size: 20px;
                        color: #333;
                        font-weight: bold;
                        background-color: transparent;
                        padding: 10px;
                        text-align: center;
                    }
                    .ResultLabel {
                        font-size: 24px;
                        font-weight: bold;
                        background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #ffcccc, stop:1 #ff6666);
                        padding: 20px;
                        text-align: center;
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        margin-top: 20px;
                        qproperty-icon: url(':/icons/checkmark.png');  # Add icon
                        qproperty-iconSize: 24px 24px;
                        qproperty-alignment: AlignCenter;
                        animation: glow 1s infinite;
                    }
                    @keyframes glow {
                        0% { border-color: #4CAF50; }
                        50% { border-color: #FF5733; }
                        100% { border-color: #4CAF50; }
                    }
                """)

        shadow_effect = QGraphicsDropShadowEffect(self)
        shadow_effect.setBlurRadius(15)
        shadow_effect.setColor(QColor(0, 0, 0, 160))
        shadow_effect.setOffset(0, 0)
        self.setGraphicsEffect(shadow_effect)
        self.progress_dialog = ProgressDialog(self)  # Initialize progress_dialog

        layout = QVBoxLayout(self)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)  # Adjust spacing between images

        # Always show the original image
        if image_path:
            image_label1 = QLabel(self)
            pixmap1 = QPixmap(image_path)
            scaled_pixmap1 = pixmap1.scaledToWidth(400, Qt.SmoothTransformation)  # Scale image to a fixed width
            image_label1.setPixmap(scaled_pixmap1)
            image_label1.setAlignment(Qt.AlignCenter)  # Center-align image
            grid_layout.addWidget(image_label1, 0, 0)

            description_label1 = QLabel("Original Lung Image", self)
            description_label1.setObjectName("TextLabel")
            description_label1.setAlignment(Qt.AlignCenter)  # Center-align description
            grid_layout.addWidget(description_label1, 1, 0)

        # Add annotated image if available
        if annotated_image_path:
            image_label2 = QLabel(self)
            pixmap2 = QPixmap(annotated_image_path)
            scaled_pixmap2 = pixmap2.scaledToWidth(400, Qt.SmoothTransformation)  # Scale image to a fixed width
            image_label2.setPixmap(scaled_pixmap2)
            image_label2.setAlignment(Qt.AlignCenter)  # Center-align image
            grid_layout.addWidget(image_label2, 0, 1)

            description_label2 = QLabel("Lung Borders Annotated", self)
            description_label2.setObjectName("TextLabel")
            description_label2.setAlignment(Qt.AlignCenter)  # Center-align description
            grid_layout.addWidget(description_label2, 1, 1)

        # Add edge image if available
        if edge_image_path:
            image_label3 = QLabel(self)
            pixmap3 = QPixmap(edge_image_path)
            scaled_pixmap3 = pixmap3.scaledToWidth(400, Qt.SmoothTransformation)  # Scale image to a fixed width
            image_label3.setPixmap(scaled_pixmap3)
            image_label3.setAlignment(Qt.AlignCenter)  # Center-align image
            grid_layout.addWidget(image_label3, 0, 2)

            description_label3 = QLabel("Edges Detected", self)
            description_label3.setObjectName("TextLabel")
            description_label3.setAlignment(Qt.AlignCenter)  # Center-align description
            grid_layout.addWidget(description_label3, 1, 2)

        layout.addLayout(grid_layout)

        # Display analysis results
        if analysis_result:
            result_text = f"Probability: {analysis_result['probability']:.6f}, Disease: {analysis_result['disease']}"
        else:
            result_text = "No analysis performed."

        self.results_label = QLabel(result_text, self)
        self.results_label.setStyleSheet("font-weight: bold;")
        self.results_label.setObjectName("ResultLabel")
        layout.addWidget(self.results_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def add_message(self, message):
        self.message_label.setText(message)


class AnalysisResultDialog(QDialog):
    def __init__(self, original_image_path, labeled_image, analysis_results, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Analysis Result')
        self.setFixedSize(1200, 800)

        layout = QVBoxLayout()

        # Add shadow effect to the dialog
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.setGraphicsEffect(shadow)

        # Display Original Image
        original_label = QLabel(self)
        original_image = QPixmap(original_image_path)
        original_label.setPixmap(original_image.scaled(550, 550, aspectRatioMode=1))
        original_label.setAlignment(Qt.AlignCenter)

        # Display Labeled Image
        labeled_label = QLabel(self)
        labeled_image = QImage(labeled_image.data, labeled_image.shape[1], labeled_image.shape[0],
                               labeled_image.strides[0], QImage.Format_RGB888)
        labeled_pixmap = QPixmap.fromImage(labeled_image)
        labeled_label.setPixmap(labeled_pixmap.scaled(550, 550, aspectRatioMode=1))
        labeled_label.setAlignment(Qt.AlignCenter)

        # Display Analysis Results
        result_text = QLabel(self)
        result_text.setText(f"Analysis Results:\n")
        for result in analysis_results:
            result_text.setText(
                result_text.text() + f"{result['disease']} with confidence {result['confidence']:.2f}\n")
        result_text.setAlignment(Qt.AlignCenter)

        # Arrange the layout
        images_layout = QHBoxLayout()
        images_layout.addWidget(original_label)
        images_layout.addWidget(labeled_label)

        layout.addLayout(images_layout)
        layout.addWidget(result_text)

        self.setLayout(layout)

        # Add a Close button
        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)


class MedicalApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Lung Health Analysis Tool')
        self.setFixedSize(900, 650)

        self.setWindowIcon(QIcon('lungs.png'))  # Add an icon for the application

        # Add a shadow effect to the main window
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.setGraphicsEffect(shadow)
        self.progress_dialog = None  # Initialize progress_dialog
        self.db_path = None  # Initialize database path
        self.analysis_options = {
            'apply_noise_reduction': True,
            'show_edges': True,
            'annotate_borders': True,
            'select_roi': False,
            'enhance_contrast': True,
            'image_scaling': 100,
            'model_selection': 'Model A',
            'confidence_threshold': 50,
            'brightness': 0,
            'contrast': 0,
            'edge_detection_sensitivity': 5
        }  # Initialize analysis options with default values
        self.analysis_options2 = {  # parameters specifically for "Set Analysis Parameters"
            'omit_resize': False,
            'omit_normalization': False,
            'omit_edge_detection': False,
            'omit_annotation': False,
            'brightness': 50,
            'contrast': 50,
            'noise_reduction': False
        }  # Initialize analysis options with default values
        self.analysis_parameters = {
            'enhance_contrast': True,
            'brightness': 0,
            'contrast': 0
        }
        self.create_menu_bar()
        self.init_ui()

        # Load the model
        model_path = 'runs/train/exp6/weights/best.pt'
        try:
            self.model = DetectMultiBackend(weights=model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device).eval()
            print("YOLOv5 model loaded successfully.")
        except Exception as e:
            self.model = None
            print(f"Failed to load YOLOv5 model: {e}")
            self.show_error_message(f"Failed to load YOLOv5 model: {e}")

    def update_image_display(self, image):
        try:
            self.current_image = image
            q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
            self.image_label.setScaledContents(True)
        except Exception as e:
            print(f"Failed to update image display: {e}")
            self.show_error_message(f"Failed to update image display: {e}")

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #34495E;
                color: white;
                font-size: 20px;
            }
            QMenuBar::item:selected {
                background-color: #1ABC9C;
            }
            QMenu {
                background-color: #34495E;
                color: white;
                border: none;
            }
            QMenu::item:selected {
                background-color: #1ABC9C;
            }
            QMenu::item {
                padding: 10px 20px;
            }
        """)

        # File menu
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction('Open Image', self.open_image)
        file_menu.addAction('Create DB', self.create_db)
        file_menu.addAction('Load DB', self.load_db)
        file_menu.addAction('Save DB', self.save_db)
        file_menu.addAction('Add Image to DB', self.add_image_to_db)
        file_menu.addAction('Add Image to DB-prof', self.add_image_to_dbb)
        file_menu.addAction('Remove Image from DB', self.remove_image_from_db)

        # View menu
        view_menu = menu_bar.addMenu('&View')
        view_menu.addAction('View Project Description', self.view_project_description)
        view_menu.addAction('Options of the Analysis', self.view_analysis_options)
        view_menu.addAction('Data Input/Output Format', self.view_data_format)

        view_menu.addAction('View Results', self.view_results)

        # Test menu
        test_menu = menu_bar.addMenu('&Test')
        test_menu.addAction('Quality Control', self.open_image_for_quality_control)

        # Analyze menu
        analyze_menu = menu_bar.addMenu('&Analyze')
        analyze_menu.addAction('Set Analysis Parameters', self.set_new_analysis_parameters)  # New function
        analyze_menu.addAction('Perform Data Analysis', self.perform_new_data_analysis)  # New function

        # Help menu
        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction('Help Options', self.show_help_dialog)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.setStyleSheet("background-color: #2C3E50;")

        vbox_layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        vbox_layout.addWidget(self.image_label)

        welcome_text = """
        <h1 style="color: #ECF0F1; font-family: 'Arial'; font-size: 28px;">
            Welcome to the Smoking-Associated Lung Damage Estimation Tool
        </h1>
        """
        welcome_label = QLabel(welcome_text, self)
        welcome_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        welcome_label.setWordWrap(True)
        vbox_layout.addWidget(welcome_label)
        #icon for main window
        gif_label = QLabel(self)
        movie = QMovie('pneumonia.gif')  # Replace with the path to your GIF file
        gif_label.setMovie(movie)

        movie.start()  # Start the GIF animation

        # Add the gif_label to your layout
        vbox_layout.addWidget(gif_label)
        separator = QLabel("<hr style='border: 1px solid #ECF0F1; width: 80%;'>")
        separator.setAlignment(Qt.AlignCenter)
        vbox_layout.addWidget(gif_label, alignment=Qt.AlignCenter)


        paragraph_text = """
        <p style="font-size: 20px; font-family: 'Arial'; color: #ECF0F1;">
            Our state-of-the-art application utilizes advanced image analysis techniques 
            to estimate lung damage caused by smoking. Harness the power of AI and deep learning 
            to gain insights into lung health.
        </p>
        <p style="font-size: 20px; font-family: 'Arial'; color: #ECF0F1;">
            Load your lung images and get a comprehensive analysis with just a few clicks.
        </p>
        """
        paragraph_label = QLabel(paragraph_text, self)
        paragraph_label.setAlignment(Qt.AlignCenter)
        paragraph_label.setWordWrap(True)
        vbox_layout.addStretch()
        vbox_layout.addWidget(paragraph_label)
        vbox_layout.addStretch()

        button_style = """
        QPushButton {
            background-color: #1ABC9C;
            color: #ECF0F1;
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
            font-family: 'Arial';
        }
        QPushButton:hover {
            background-color: #16A085;
        }
        """

        open_image_button = QPushButton('Open Image', self)
        open_image_button.setStyleSheet(button_style)
        open_image_button.clicked.connect(self.open_image)
        vbox_layout.addWidget(open_image_button)

        vbox_layout.addStretch()

    def open_image(self):
        print("Opening image")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            print(f"Image selected: {file_path}")
            try:
                self.analyze_image(file_path)
                # Remove or comment out the following line
                # self.update_image_display(self.current_image)
            except Exception as e:
                print(f"Error in open_image: {e}")
                self.show_error_message(f"Error opening image: {e}")

    def show_histogram_input_1(self):
        print("show_histogram_input_1 called")
        normal_count = 50
        pneumonia_count = 70

        # Create a new MatplotlibCanvas and embed it in the window
        canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        canvas.plot_histogram(normal_count, pneumonia_count)

        # Display the canvas in a new dialog or main window
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.setWindowTitle("Histogram of Lung Conditions")
        dialog.exec_()  # Show the dialog with the plot

    def show_histogram_input_2(self):
        normal_count = 87
        pneumonia_count = 42
        # Create a new MatplotlibCanvas and embed it in the window
        canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        canvas.plot_histogram(normal_count, pneumonia_count)

        # Display the canvas in a new dialog or main window
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.setWindowTitle("Histogram of Lung Conditions")
        dialog.exec_()  # Show the dialog with the plot
    #Test ~ Quality Control function is here


    def is_image_suitable(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return False  # Cannot load image

        # Convert the image to grayscale (since X-ray images should be in this format)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check resolution (example: minimum 256x256, adjust as needed)
        height, width = gray_image.shape[:2]
        if height < 256 or width < 256:
            return False  # Image resolution is too low

        # Additional check: Ensure the image isn't completely black or white
        if np.mean(gray_image) < 10 or np.mean(gray_image) > 245:
            return False  # Image is too dark or too bright

        return True  # Image passed all checks

    def check_image_quality(self, image_path):
        # Placeholder function to check image quality
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False, "Unable to load image."

        # Check for sufficient contrast
        min_val, max_val, _, _ = cv2.minMaxLoc(image)
        if max_val - min_val < 50:  # Arbitrary threshold for demonstration
            return False, "Insufficient contrast."

        # Check for adequate brightness
        mean_val = np.mean(image)
        if mean_val < 30 or mean_val > 225:  # Arbitrary brightness thresholds
            return False, "Image brightness is not optimal."

        return True, "All checks passed."
    def show_histogram_input_3(self):
        normal_count = 45
        pneumonia_count = 41
        # Create a new MatplotlibCanvas and embed it in the window
        canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        canvas.plot_histogram(normal_count, pneumonia_count)

        # Display the canvas in a new dialog or main window
        dialog = QDialog(self)
        layout = QVBoxLayout(dialog)
        layout.addWidget(canvas)
        dialog.setWindowTitle("Histogram of Lung Conditions")
        dialog.exec_()  # Show the dialog with the plot

    def show_help_dialog(self):
        help_text = """
            <h2 style="color: white; font-family: 'Arial'; font-size: 28px; text-align: center;">Help Guide</h2>
            <h3 style="color: white; font-family: 'Arial'; font-size: 20px;">Welcome to the Smoking-Associated Lung Damage Estimation Tool. This guide will help you understand how to use the application.</h3>
            <h3 style="color: white; font-family: 'Arial'; font-size: 20px;">Main Features</h3>
            <ul style="color: white;">
                <li><strong>Open Image:</strong> Load an image for analysis.</li>
                <li><strong>Create DB:</strong> Create a new database to store images and results.</li>
                <li><strong>Load DB:</strong> Load an existing database.</li>
                <li><strong>Save DB:</strong> Save the current database.</li>
                <li><strong>Add Image to DB:</strong> Add an image to the current database.</li>
                <li><strong>Remove Image from DB:</strong> Remove an image from the current database.</li>
                <li><strong>View Project Description:</strong> View the project description and objectives.</li>
                <li><strong>Options of the Analysis:</strong> Configure analysis options.</li>
                <li><strong>Data Input/Output Format:</strong> View the data input and output format.</li>
                <li><strong>View Results:</strong> View analysis results.</li>
                <li><strong>Quality Control:</strong> Perform quality control checks on an image.</li>
                <li><strong>Perform Data Analysis:</strong> Analyze the loaded image.</li>
            </ul>
            <h3 style="color: white; font-family: 'Arial'; font-size: 20px;">Using the Application</h3>
            <h3 style="color: white; font-family: 'Arial'; font-size: 16px;">To get started, use the 'Open Image' option to load an image for analysis. You can then configure the analysis options through 'Options of the Analysis' and perform the analysis using the 'Perform Data Analysis' option.</h3>
            <h2 style="color: white; font-family: 'Arial'; font-size: 28px;">Contact</h2>
            <h3 style="color: white; font-family: 'Arial'; font-size: 16px;">If you have any questions or need further assistance, please contact lhasupport@lhacompany.com.</h3>
        """

        dialog = QDialog(self)
        dialog.setWindowTitle("Help")

        layout = QVBoxLayout(dialog)

        # Apply a more modern and designed stylesheet
        dialog.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E
                );
                border-radius: 20px;
                padding: 15px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)

        # Add a shadow effect for depth
        shadow = QGraphicsDropShadowEffect(dialog)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 150))
        dialog.setGraphicsEffect(shadow)

        label = QLabel(help_text, dialog)
        label.setTextFormat(Qt.RichText)  # Use RichText to apply HTML styling
        label.setWordWrap(True)  # Enable word wrapping for long text
        layout.addWidget(label)

        # Add a Close button with some padding for better spacing
        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec_()

    def noise_reduce_image(self, image_path):
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load the image for noise reduction.")

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply noise reduction
            noise_reduced_image = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)

            return noise_reduced_image
        except Exception as e:
            print(f"Failed to reduce noise in image: {e}")
            return None

    def quality_control_action(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File for Quality Control", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            print(f"Selected file: {file_path}")
            analysis_successful = self.quality_control(file_path)
            if analysis_successful:
                print("Analysis successful, displaying results.")
                professional_result = self.fetch_professional_results(file_path)
                manual_result = self.get_manual_input()
                analysis_result = self.analyze_image_for_quality_control(file_path)  # hani
                combined_results = {
                    'professional': professional_result,
                    'manual': manual_result,
                    'automatic': analysis_result
                }
                if combined_results:
                    self.show_quality_control_results(file_path, combined_results)
                else:
                    print("Quality control or analysis failed.")
                    self.show_error_message("Quality control failed or image analysis failed.")
            else:
                print("Quality control or analysis failed.")
                self.show_error_message("Quality control failed or image analysis failed.")

    def set_new_analysis_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Analysis Parameters")

        layout = QVBoxLayout(dialog)
        dialog.setStyleSheet("""
                QLabel, QCheckBox, QSlider {
                    color: white;
                    font-size: 16px;
                    font-family: 'Arial';
                }
                QDialog {
                    background-color: #34495E;
                    border-radius: 15px;
                    padding: 20px;
                }
                QPushButton {
                    background-color: #1ABC9C;
                    color: white;
                    font-size: 16px;
                    font-family: 'Arial';
                    border: none;
                    padding: 10px 20px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #16A085;
                }
            """)
        # Add checkboxes for omitting different parts of the analysis
        omit_resize_checkbox = QCheckBox("Omit image resizing")
        omit_resize_checkbox.setChecked(self.analysis_parameters.get('omit_resize', False))
        layout.addWidget(omit_resize_checkbox)

        omit_normalization_checkbox = QCheckBox("Omit image normalization")
        omit_normalization_checkbox.setChecked(self.analysis_parameters.get('omit_normalization', False))
        layout.addWidget(omit_normalization_checkbox)

        omit_edge_detection_checkbox = QCheckBox("Omit edge detection")
        omit_edge_detection_checkbox.setChecked(self.analysis_parameters.get('omit_edge_detection', False))
        layout.addWidget(omit_edge_detection_checkbox)

        omit_annotation_checkbox = QCheckBox("Omit annotation")
        omit_annotation_checkbox.setChecked(self.analysis_parameters.get('omit_annotation', False))
        layout.addWidget(omit_annotation_checkbox)

        # Brightness and Contrast sliders
        layout.addWidget(QLabel("Brightness Adjustment:"))
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setMinimum(-100)
        brightness_slider.setMaximum(100)
        brightness_slider.setValue(self.analysis_parameters.get('brightness', 0))
        layout.addWidget(brightness_slider)

        layout.addWidget(QLabel("Contrast Adjustment:"))
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setMinimum(-100)
        contrast_slider.setMaximum(100)
        contrast_slider.setValue(self.analysis_parameters.get('contrast', 0))
        layout.addWidget(contrast_slider)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def save_parameters():
            # Store all the selected parameters
            self.analysis_parameters['omit_resize'] = omit_resize_checkbox.isChecked()
            self.analysis_parameters['omit_normalization'] = omit_normalization_checkbox.isChecked()
            self.analysis_parameters['omit_edge_detection'] = omit_edge_detection_checkbox.isChecked()
            self.analysis_parameters['omit_annotation'] = omit_annotation_checkbox.isChecked()
            self.analysis_parameters['brightness'] = brightness_slider.value()
            self.analysis_parameters['contrast'] = contrast_slider.value()

            print(f"Saved Parameters: {self.analysis_parameters}")
            dialog.accept()

        button_box.accepted.connect(save_parameters)
        button_box.rejected.connect(dialog.reject)

        dialog.setLayout(layout)
        dialog.exec_()

    def perform_new_data_analysis(self):
        print("Performing data analysis...")

        # Open a file dialog to let the user select an image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)

        if not file_path:
            print("No image selected.")
            return

        print(f"Selected image: {file_path}")

        # Load the image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            return

        try:
            # Step 1: Resize (if not omitted)
            if not self.analysis_parameters.get('omit_resize', False):
                image = cv2.resize(image, (256, 256))  # Example resize to 256x256
                print("Image resized.")

            # Step 2: Normalize (if not omitted)
            if not self.analysis_parameters.get('omit_normalization', False):
                image = image / 255.0  # Normalize to [0, 1] range
                print("Image normalized.")

                # Convert back to 8-bit unsigned integer (CV_8U)
                image = (image * 255).astype('uint8')
                print("Image converted back to 8-bit format.")

            # Step 3: Apply Brightness and Contrast Adjustment (if not omitted)
            if not self.analysis_parameters.get('omit_brightness_contrast', False):
                brightness = self.analysis_parameters.get('brightness', 0)
                contrast = self.analysis_parameters.get('contrast', 0)
                alpha = contrast / 100.0 + 1  # Contrast control
                beta = brightness  # Brightness control
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                print(f"Brightness: {brightness}, Contrast: {contrast} applied.")

            # Step 4: Noise Reduction (if enabled)
            if self.analysis_parameters.get('noise_reduction', False):
                image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                print("Noise reduction applied.")

            # Convert the image to RGB format for consistency with your YOLOv5 model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 5: Save the adjusted image temporarily to pass it as a file path (YOLOv5 expects file path)
            temp_image_path = "temp_adjusted_image.png"
            cv2.imwrite(temp_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            # Call analyze_image with the adjusted image file path
            analysis_result = self.analyze_image_analysis(temp_image_path)

            if analysis_result is None:
                print("Analysis returned no results.")
                return

        except Exception as e:
            print(f"Error during data analysis: {e}")
            self.show_error_message(f"Error during data analysis: {e}")

        # If the analysis is successful, continue here (result already displayed in analyze_image)
        try:
            # Display the adjusted image in the result dialog
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert OpenCV image to QImage format for displaying in QLabel
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Create a result dialog window to display the image and analysis result
            result_dialog = QDialog(self)
            result_dialog.setWindowTitle("Analysis Results")

            layout = QVBoxLayout(result_dialog)

            # Display the final image with adjustments
            image_label = QLabel()
            image_label.setPixmap(QPixmap.fromImage(qimage).scaled(512, 512, Qt.KeepAspectRatio))
            layout.addWidget(image_label)

            # Add a label for the analysis result (you can modify this as needed)
            result_label = QLabel("The analysis result has been displayed.")
            result_label.setWordWrap(True)  # Allow the text to wrap if too long
            layout.addWidget(result_label)

            # Add a close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(result_dialog.accept)
            layout.addWidget(close_button)

            result_dialog.setLayout(layout)
            result_dialog.exec_()

        except Exception as e:
            print(f"Error displaying analysis result: {e}")
            self.show_error_message(f"Error displaying analysis result: {e}")

    def show_quality_control_results(self, image_path, analysis_result=None):
        print(f"Showing quality control results for image: {image_path}")

        # Generate a random automatic result
        diseases = ['Normal', 'Pneumonia']  # List of possible diseases
        automatic_disease = random.choice(diseases)  # Randomly choose a disease
        automatic_probability = random.uniform(0.01, 0.99)  # Generate a random probability between 0.01 and 0.99

        if analysis_result:
            manual_result = analysis_result.get('manual', {})
            professional_result = analysis_result.get('professional', {})

            manual_disease = manual_result.get('disease', 'Unknown')
            manual_notes = manual_result.get('notes', 'No notes')

            professional_disease = professional_result.get('disease', 'Unknown')
            professional_notes = professional_result.get('notes', 'No professional results found')

            result_text = f"Automatic Prediction: {automatic_disease} with probability {automatic_probability:.2f}\n"
            result_text += f"Manual Annotation: {manual_disease} - {manual_notes}\n"
            result_text += f"Professional Annotation: {professional_disease} - {professional_notes}"
        else:
            result_text = "No analysis performed."

        # Setup the dialog window
        dialog = QDialog(self)
        dialog.setWindowTitle('Quality Control Results')
        dialog.setGeometry(100, 100, 1200, 600)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
                border-radius: 15px;
                border: 2px solid #ccc;
            }
            QLabel {
                font-size: 18px;
                color: #333;
                padding: 10px;
                background-color: transparent;
                border: none;
            }
            .TextLabel {
                font-size: 20px;
                color: #333;
                font-weight: bold;
                background-color: transparent;
                padding: 10px;
                text-align: center;
            }
            .ResultLabel {
                font-size: 24px;
                color: #4CAF50;
                font-weight: bold;
                background-color: transparent;
                padding: 20px;
                text-align: center;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                margin-top: 20px;
            }
        """)

        layout = QVBoxLayout(dialog)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)

        # Display the original lung image
        if image_path:
            image_label = QLabel(self)
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            grid_layout.addWidget(image_label, 0, 0)

            description_label = QLabel("Original Lung Image", self)
            description_label.setObjectName("TextLabel")
            description_label.setAlignment(Qt.AlignCenter)
            grid_layout.addWidget(description_label, 1, 0)

        layout.addLayout(grid_layout)

        results_label = QLabel(result_text, self)
        results_label.setObjectName("ResultLabel")
        layout.addWidget(results_label, alignment=Qt.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec_()

    #############

    #############

    def display_analysis_results(self, analysis_result):
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Results")
        layout = QVBoxLayout(dialog)

        # Display basic analysis results
        result_text = f"Probability: {analysis_result['probability']:.2f}\nDisease: {analysis_result['disease']}"
        result_label = QLabel(result_text, self)
        layout.addWidget(result_label)

        # Display additional images based on user selections
        images = analysis_result.get('images', {})
        for key, img in images.items():
            label = QLabel(self)
            pixmap = QPixmap.fromImage(
                QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888))
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(QLabel(f"{key.capitalize()} Image:"))
            layout.addWidget(label)

        # Add dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

    def fetch_professional_results(self, image_path):
        try:
            with sqlite3.connect('lung_images_by_doctors.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT professional_result FROM images WHERE path = ?', (image_path,))
                result = cursor.fetchone()
                if result:
                    return {'disease': result[0], 'notes': 'Professional annotation'}
                return {'disease': 'Unknown', 'notes': 'No professional results found'}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {e}")
            return {'disease': 'Error', 'notes': str(e)}

    def analyze_image(self, image_path):
        print("Analyzing image with YOLOv5 model.")
        if not self.model:
            self.show_error_message("YOLOv5 model not loaded.")
            return None

        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not loaded properly")

            print(f"Original image shape: {image.shape}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply the scaling factor
            scaling_factor = self.analysis_options.get('image_scaling', 100) / 100.0
            image_rgb = cv2.resize(image_rgb, None, fx=scaling_factor, fy=scaling_factor,
                                   interpolation=cv2.INTER_LINEAR)
            print(f"Scaled image shape: {image_rgb.shape}")

            # Apply noise reduction if enabled
            if self.analysis_options.get('noise_reduction', False):
                image_rgb = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)

            # Apply brightness and contrast adjustments if enabled
            if self.analysis_options.get('enhance_contrast', False):
                brightness = self.analysis_options.get('brightness', 0)
                contrast = self.analysis_options.get('contrast', 0)
                alpha = contrast / 100.0 + 1  # Contrast control
                beta = brightness  # Brightness control
                image_rgb = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)

            # Generate edges using Canny edge detection
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(image_gray, 50, 150)

            # Detect lung borders using contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lung_borders = image_rgb.copy()
            cv2.drawContours(lung_borders, contours, -1, (255, 165, 0), 2)  # Change to orange (BGR format)

            # Prepare the image for YOLOv5 model
            img_resized = cv2.resize(image_rgb, (640, 640))
            img_normalized = img_resized / 255.0
            img_transposed = img_normalized.transpose((2, 0, 1))  # Convert HWC to CHW format
            img_tensor = torch.from_numpy(img_transposed).float().unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                pred = self.model(img_tensor)
                pred = non_max_suppression(pred, conf_thres=0.25)

            # Check if no lungs were detected
            if pred[0] is None or len(pred[0]) == 0:
                self.show_error_message("No lungs detected. The image may not be a valid lung X-ray.")
                return None

            # Process and analyze the output
            detection_results = []
            labeled_image = image_rgb.copy()
            for detection in pred[0]:
                class_id = int(detection[5].cpu().numpy())
                confidence = float(detection[4].cpu().numpy())
                x1, y1, x2, y2 = map(int, detection[:4].cpu().numpy())
                if class_id == 0:
                    disease = "Normal"
                    color = (0, 255, 0)  # Green for normal
                elif class_id == 1:
                    disease = "Pneumonia"
                    color = (255, 0, 0)  # Red for pneumonia
                else:
                    disease = "Unknown"
                    color = (255, 255, 255)  # White for unknown

                # Draw the bounding box and label
                cv2.rectangle(labeled_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(labeled_image, f'{disease} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

                detection_results.append({
                    'disease': disease,
                    'confidence': confidence
                })

            # Show the results in the custom dialog, including the edges and lung borders image
            self.show_analysis_results(image_path, labeled_image, edges, lung_borders, detection_results)

        except Exception as e:
            print(f"Failed to analyze image: {e}")
            self.show_error_message(f"Failed to analyze image: {e}")
            return None

    def analyze_image_analysis(self, image_path):
        print("Analyzing image with YOLOv5 model.")
        if not self.model:
            self.show_error_message("YOLOv5 model not loaded.")
            return None

        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not loaded properly")

            print(f"Original image shape: {image.shape}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply the scaling factor
            scaling_factor = self.analysis_options.get('image_scaling', 100) / 100.0
            image_rgb = cv2.resize(image_rgb, None, fx=scaling_factor, fy=scaling_factor,
                                   interpolation=cv2.INTER_LINEAR)
            print(f"Scaled image shape: {image_rgb.shape}")

            # Apply noise reduction if enabled
            if self.analysis_options2.get('noise_reduction', False):
                image_rgb = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)

            # Apply brightness and contrast adjustments if enabled
            if self.analysis_parameters.get('enhance_contrast', False):
                brightness = self.analysis_parameters.get('brightness', 0)
                contrast = self.analysis_parameters.get('contrast', 0)
                alpha = contrast / 100.0 + 1  # Contrast control
                beta = brightness  # Brightness control
                image_rgb = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)

            # Generate edges using Canny edge detection
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(image_gray, 50, 150)

            # Detect lung borders using contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lung_borders = image_rgb.copy()
            cv2.drawContours(lung_borders, contours, -1, (255, 165, 0), 2)  # Change to orange (BGR format)

            # Prepare the image for YOLOv5 model
            img_resized = cv2.resize(image_rgb, (640, 640))
            img_normalized = img_resized / 255.0
            img_transposed = img_normalized.transpose((2, 0, 1))  # Convert HWC to CHW format
            img_tensor = torch.from_numpy(img_transposed).float().unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                pred = self.model(img_tensor)
                pred = non_max_suppression(pred, conf_thres=0.25)

            # Check if no lungs were detected
            if pred[0] is None or len(pred[0]) == 0:
                self.show_error_message("No lungs detected. The image may not be a valid lung X-ray.")
                return None

            # Process and analyze the output
            detection_results = []
            labeled_image = image_rgb.copy()
            for detection in pred[0]:
                class_id = int(detection[5].cpu().numpy())
                confidence = float(detection[4].cpu().numpy())
                x1, y1, x2, y2 = map(int, detection[:4].cpu().numpy())
                if class_id == 0:
                    disease = "Normal"
                    color = (0, 255, 0)  # Green for normal
                elif class_id == 1:
                    disease = "Pneumonia"
                    color = (255, 0, 0)  # Red for pneumonia
                else:
                    disease = "Unknown"
                    color = (255, 255, 255)  # White for unknown

                # Draw the bounding box and label
                cv2.rectangle(labeled_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(labeled_image, f'{disease} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

                detection_results.append({
                    'disease': disease,
                    'confidence': confidence
                })

            # Show the results in the custom dialog, including the edges and lung borders image
            self.show_analysis_results_analysisMenu(image_path, labeled_image, edges, lung_borders, detection_results)

        except Exception as e:
            print(f"Failed to analyze image: {e}")
            self.show_error_message(f"Failed to analyze image: {e}")
            return None

    def show_analysis_results(self, original_image_path, labeled_image, edges_image, lung_borders_image,
                              analysis_results):
        # Apply noise reduction if enabled
        if self.analysis_options.get('noise_reduction', False):
            noise_reduced_image = self.noise_reduce_image(original_image_path)
        else:
            noise_reduced_image = None

        # Only pass the images that should be shown based on the analysis options
        if not self.analysis_options['show_edges']:
            edges_image = None
        if not self.analysis_options['annotate_borders']:
            lung_borders_image = None


        dialog = AnalysisResultDialog(original_image_path, labeled_image, edges_image, lung_borders_image,
                                      noise_reduced_image, analysis_results, self)
        dialog.exec_()


    def show_analysis_results_analysisMenu(self, original_image_path, labeled_image, edges_image, lung_borders_image,
                              analysis_results):
        # Apply noise reduction if enabled
        if self.analysis_options2.get('noise_reduction', False):
            noise_reduced_image = self.noise_reduce_image(original_image_path)
        else:
            noise_reduced_image = None

        # Only pass the images that should be shown based on the analysis options
        if not self.analysis_options['show_edges']:
            edges_image = None
        if not self.analysis_options['annotate_borders']:
            lung_borders_image = None

        dialog = AnalysisResultDialog(original_image_path, labeled_image, edges_image, lung_borders_image,
                                      noise_reduced_image, analysis_results, self)
        dialog.exec_()

    def open_image_for_quality_control(self):
        print("Opening image for quality control")
        db_path = os.path.join(os.getcwd(), "lung_images_by_doctors.db")
        self.db_path = db_path

        if not os.path.exists(db_path):
            self.show_error_message(f"Database file not found: {db_path}")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File for Quality Control", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            print(f"Image selected for quality control: {file_path}")
            try:
                self.ensure_table_exists(db_path)
                professional_result = self.fetch_professional_results(file_path)
                manual_result = self.get_manual_input()
                automatic_result = self.analyze_image_for_quality_control(file_path)

                combined_results = {
                    'professional': professional_result,
                    'manual': manual_result,
                    'automatic': automatic_result
                }

                if combined_results:
                    self.show_quality_control_results(file_path, combined_results)
                else:
                    print("Quality control or analysis failed.")
                    self.show_error_message("Quality control failed or image analysis failed.")
            except Exception as e:
                print(f"Error in open_image_for_quality_control: {e}")
                self.show_error_message(f"Error opening image for quality control: {e}")

    def ensure_table_exists(self, db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT UNIQUE,
                        professional_result TEXT
                    )
                ''')
                conn.commit()
            print("Table ensured to exist.")
        except Exception as e:
            print(f"Failed to ensure table exists: {e}")
            self.show_error_message(f"Failed to ensure table exists: {e}")

    def analyze_image_for_quality_control(self, image_path):
        print("Analyzing image for Quality Control with YOLOv5 model.")
        if not self.model:
            print("Error: YOLOv5 model not loaded.")
            return "Error: Model not loaded"

        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not loaded properly")

            print(f"Original image shape: {image.shape}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 640))  # Resize to model expected size

            image = image.transpose((2, 0, 1))  # Convert HWC to CHW format
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                pred = self.model(image_tensor)
                pred = non_max_suppression(pred, conf_thres=0.25)

            # Check if no lungs were detected
            if pred[0] is None or len(pred[0]) == 0:
                return "No lungs detected"

            # Analyze the output to determine the most likely diagnosis
            max_confidence = 0
            diagnosis = "Unknown"
            for detection in pred[0]:
                class_id = int(detection[5].cpu().numpy())
                confidence = float(detection[4].cpu().numpy())
                if confidence > max_confidence:
                    if class_id == 0:
                        diagnosis = "Normal"
                    elif class_id == 1:
                        diagnosis = "Pneumonia"
                    max_confidence = confidence

            return diagnosis

        except Exception as e:
            print(f"Failed to analyze image: {e}")
            return f"Error: {str(e)}"

    def fetch_professional_results(self, image_path):
        normalized_path = os.path.abspath(image_path).replace("\\", "/").strip().lower()  # Normalize the path
        print("Normalized query path:", normalized_path)  # Debug output to verify path
        try:
            with sqlite3.connect('lung_images_by_doctors.db') as conn:
                cursor = conn.cursor()
                query = 'SELECT professional_result FROM images WHERE lower(trim(path)) = ?'
                cursor.execute(query, (normalized_path,))
                result = cursor.fetchone()
                if result:
                    return {'disease': result[0], 'notes': 'Professional annotation'}
                else:
                    return {'disease': 'Unknown', 'notes': 'No professional results found'}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {e}")
            return {'disease': 'Error', 'notes': str(e)}

    def prompt_for_image_and_compare(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image for Comparison", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if not file_path:  # Check if the file path is empty
            QMessageBox.warning(self, "Warning", "No image was selected.")
            return

        self.show_comparison_results(file_path)

    def show_comparison_results(self, image_path):
        if not image_path:
            QMessageBox.critical(self, "Error", "No image selected.")
            return

        # Fetch results
        auto_results = self.analyze_image(image_path)
        manual_results = self.get_manual_input()
        prof_results = self.fetch_professional_results(image_path)

        # Display in a QDialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Comparison of Results")
        dialog.resize(500, 400)

        layout = QVBoxLayout()

        # Displaying each result with labels
        auto_label = QLabel(
            f"Automatic Result: {auto_results['disease']} with confidence {auto_results['confidence']:.2f}")
        manual_label = QLabel(f"Manual Result: {manual_results['disease']}")
        prof_label = QLabel(f"Professional Result: {prof_results['disease']} ({prof_results['notes']})")

        layout.addWidget(auto_label)
        layout.addWidget(manual_label)
        layout.addWidget(prof_label)

        dialog.setLayout(layout)
        dialog.exec_()

    def get_manual_input(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Input")
        dialog.setFixedSize(300, 200)
        layout = QVBoxLayout()

        label = QLabel("Enter your assessment (e.g., Normal or Pneumonia):")
        layout.addWidget(label)

        manual_input = QLineEdit(dialog)
        layout.addWidget(manual_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def on_accept():
            self.manual_result = {'disease': manual_input.text(), 'notes': 'Manual annotation'}
            dialog.accept()

        button_box.accepted.connect(on_accept)
        button_box.rejected.connect(dialog.reject)

        dialog.setLayout(layout)
        dialog.exec_()

        return self.manual_result if hasattr(self, 'manual_result') else {'disease': 'Unknown',
                                                                          'notes': 'No manual input provided'}

    def fetch_manual_results(self, image_path):
        # Fetch manual results from the database (mock implementation)
        return {'disease': 'Normal', 'notes': 'Manual annotation: normal lungs'}

    def insert_professional_result(self, image_path, professional_result):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO images (path, professional_result)
                    VALUES (?, ?)
                ''', (image_path, professional_result))
                print("Professional result inserted successfully.")
        except Exception as e:
            print(f"Error inserting professional result: {e}")

    def add_image_to_dbb(self):
        # Open file dialog to select an image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            # Input dialog to get professional result
            professional_result, ok = QInputDialog.getText(self, 'Enter Professional Result',
                                                           'Enter the professional diagnosis result:')
            if ok:
                # Function to insert data into the database
                self.insert_image_and_result(file_path, professional_result)

    def insert_image_and_result(self, image_path, professional_result):
        try:
            with sqlite3.connect('lung_images_by_doctors.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (path, professional_result)
                    VALUES (?, ?)
                ''', (image_path, professional_result))
                conn.commit()
            QMessageBox.information(self, "Success", "Image and result added successfully.")
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Warning", "This image path already exists in the database.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
    def save_temp_image(self, image):
        import tempfile
        _, temp_filename = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(temp_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return temp_filename

    def get_manual_input(self):
        # Create a custom QDialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Diagnosis Input")
        dialog.setFixedSize(400, 200)  # Adjust the size as necessary

        layout = QVBoxLayout(dialog)

        # Apply a modern stylesheet to the dialog
        dialog.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E
                );
                border-radius: 20px;
                padding: 20px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-family: 'Arial';
            }
            QLineEdit {
                background-color: #34495E;
                color: white;
                font-size: 16px;
                padding: 5px;
                border: 1px solid white;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
            QDialogButtonBox QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #16A085;
            }
        """)

        # Add a label for the input instruction
        label = QLabel("Enter your manual diagnosis result:")
        layout.addWidget(label)

        # Add a QLineEdit for manual input
        line_edit = QLineEdit(dialog)
        layout.addWidget(line_edit)

        # Create buttons for OK and Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        layout.addWidget(buttons)

        # Handle button clicks
        def onButtonClick(button):
            if button == QDialogButtonBox.Ok:
                manual_result = line_edit.text()
                dialog.accept()
                return {'disease': manual_result if manual_result else 'No input provided'}
            else:
                dialog.reject()
                return {'disease': 'No input provided'}

        buttons.accepted.connect(lambda: onButtonClick(QDialogButtonBox.Ok))
        buttons.rejected.connect(lambda: onButtonClick(QDialogButtonBox.Cancel))

        dialog.setLayout(layout)
        result = dialog.exec_()

        # Return result based on dialog acceptance
        if result == QDialog.Accepted:
            return {'disease': line_edit.text() if line_edit.text() else 'No input provided'}
        else:
            return {'disease': 'No input provided'}

    def get_model_path(self, model_selection):
        model_paths = {
            "Model A": "models/model.h5",
            "Model B": "models/model_b.h5",
            "Model C": "models/model_c.h5"
        }
        return model_paths.get(model_selection, "models/model.h5")

    def get_model_path(self, model_selection):
        model_paths = {
            "Model A": "models/model.h5",
            "Model B": "models/model2.h5",
            "Model C": "models/model_c.h5"
        }
        return model_paths.get(model_selection, "models/model.h5")

    def detect_edges(self, image_path):
        try:
            # Load the image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Retrieve the sensitivity value from analysis options
            edge_sensitivity = self.analysis_options.get('edge_sensitivity', 5)

            # Apply Canny edge detection with sensitivity settings
            lower_threshold = edge_sensitivity * 10
            upper_threshold = lower_threshold * 2
            edges = cv2.Canny(img, lower_threshold, upper_threshold)

            edge_image_path = 'temp_edges.png'
            cv2.imwrite(edge_image_path, edges)

            return edge_image_path
        except Exception as e:
            self.show_error_message(f"Failed to detect edges: {e}")
            return None

    def annotate_lung_borders(self, image_path, is_pneumonia):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load the image for annotation.")

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Use binary thresholding with Otsu's method to segment the lungs
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use morphological operations to clean up the segmented image
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the cleaned binary image
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to focus on lung-like shapes based on area
        lung_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Focus on contours that are likely to be lungs
            if 10000 < area < 300000:  # These values may need to be adjusted based on your image characteristics
                lung_contours.append(contour)

        # Check if lung contours are found
        if not lung_contours:
            print("No lung contours found.")
            return image_path

        # Draw the lung contours on the original image
        annotation_color = (0, 255, 0) if not is_pneumonia else (0, 0, 255)  # Green for normal, red for pneumonia
        cv2.drawContours(image, lung_contours, -1, annotation_color, 2)  # Thicker line for better visibility

        # Save the annotated image
        annotated_image_path = 'annotated_lung_borders.png'
        cv2.imwrite(annotated_image_path, image)

        return annotated_image_path

    import cv2
    import numpy as np

    def annotate_lung_borders_with_disease_arrow(self, image_path, disease_location=None, is_pneumonia=False):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load the image for annotation.")

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Use binary thresholding with Otsu's method to segment the lungs
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use morphological operations to clean up the segmented image
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the cleaned binary image
        contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to focus on lung-like shapes based on area
        lung_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 300000:  # Adjusted area thresholds
                lung_contours.append(contour)

        # Check if lung contours are found
        if not lung_contours:
            print("No lung contours found.")
            return image_path

        # Draw the lung contours on the original image
        annotation_color = (0, 255, 0) if not is_pneumonia else (0, 0, 255)  # Green for normal, red for pneumonia
        cv2.drawContours(image, lung_contours, -1, annotation_color, 2)  # Thicker line for better visibility

        # If a disease location is specified, draw an arrow pointing to it
        if disease_location:
            start_point = (disease_location[0] - 50, disease_location[1] - 50)  # Adjust the start point as needed
            end_point = disease_location
            arrow_color = (255, 0, 0)  # Blue color for the arrow
            cv2.arrowedLine(image, start_point, end_point, arrow_color, 3, tipLength=0.5)

        # Save the annotated image
        annotated_image_path = 'annotated_lung_with_disease.png'
        cv2.imwrite(annotated_image_path, image)

        return annotated_image_path

    def create_db(self):
        # Create a custom QDialog for creating the database
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Database")
        dialog.setFixedSize(400, 200)

        layout = QVBoxLayout(dialog)

        # Apply a modern stylesheet to the dialog
        dialog.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E
                );
                border-radius: 20px;
                padding: 20px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-family: 'Arial';
            }
            QLineEdit {
                background-color: #34495E;
                color: white;
                font-size: 16px;
                padding: 5px;
                border: 1px solid white;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
            QDialogButtonBox QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #16A085;
            }
        """)

        # Add a label for the database name
        label = QLabel("Enter database name:")
        layout.addWidget(label)

        # Add a QLineEdit for database name input
        line_edit = QLineEdit(dialog)
        layout.addWidget(line_edit)

        # Create OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        # Handle button clicks
        def onButtonClick(button):
            if button == QDialogButtonBox.Ok:
                db_name = line_edit.text()
                if db_name:
                    self.db_path = os.path.join(os.getcwd(), f"{db_name}.sql")
                    try:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT)"
                            )
                            conn.commit()
                        print(f"Database '{db_name}' created and saved at {self.db_path}.")
                        # Show success message box
                        msg_box = QMessageBox()
                        msg_box.setText("Database Created")
                        msg_box.setInformativeText(f"Database '{db_name}' created and saved.")
                        msg_box.setStyleSheet("""
                            QLabel {
                                color: white;
                                font-size: 14px;
                                font-family: 'Arial';
                            }
                            QMessageBox {
                                background-color: #2C3E50;
                                border-radius: 10px;
                            }
                            QPushButton {
                                background-color: #1ABC9C;
                                color: white;
                                font-size: 14px;
                                border: none;
                                padding: 10px 15px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #16A085;
                            }
                        """)
                        msg_box.exec_()

                        dialog.accept()
                    except Exception as e:
                        print(f"Failed to create database: {e}")
                        self.show_error_message(f"Failed to create database: {e}")
                else:
                    QMessageBox.warning(self, "Input Error", "Please enter a valid database name.")
            elif button == QDialogButtonBox.Cancel:
                dialog.reject()

        buttons.accepted.connect(lambda: onButtonClick(QDialogButtonBox.Ok))
        buttons.rejected.connect(lambda: onButtonClick(QDialogButtonBox.Cancel))

        dialog.setLayout(layout)
        dialog.exec_()

    def load_professional_results(self, db_path, image_path):
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT professional_result FROM images WHERE path=?", (image_path,))
                result = cursor.fetchone()
                if result and result[0]:
                    print("Professional results loaded successfully.")
                    professional_result = {'disease': result[0], 'notes': 'Professional annotation'}
                else:
                    print("No professional results found.")
                    professional_result = {'disease': 'None', 'notes': 'No professional results found'}

                # Get manual input from the user
                manual_result = self.get_manual_input()

                self.show_quality_control_results(
                    image_path,
                    analysis_result={
                        'professional': professional_result,
                        'manual': manual_result
                    }
                )
        except Exception as e:
            print(f"Failed to fetch professional results: {e}")
            self.show_error_message(f"Failed to fetch professional results: {e}")

    def load_db(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Database File", "",
                                                   "Database Files (*.sql)", options=options)
        if file_path:
            self.db_path = file_path
            print(f"Database loaded from {self.db_path}.")
            msg_box = QMessageBox()
            msg_box.setText("Database Loaded")
            msg_box.setInformativeText(f"Database loaded from {self.db_path}.")
            msg_box.setStyleSheet("QLabel{ color: black; }")
            msg_box.exec_()

    def save_db(self):
        if not self.db_path:
            msg_box = QMessageBox()
            msg_box.setStyleSheet("QLabel { color: black; }")
            msg_box.setText("Error")
            msg_box.setInformativeText("No database loaded or created.")
            msg_box.exec_()
            return

        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Database File", "",
                                                   "Database Files (*.sql)", options=options)
        if save_path:
            try:
                os.rename(self.db_path, save_path)
                self.db_path = save_path
                print(f"Database saved at {self.db_path}.")
                msg_box = QMessageBox()
                msg_box.setText("Database Saved")
                msg_box.setInformativeText(f"Database saved at {self.db_path}.")
                msg_box.setStyleSheet("QLabel{ color: black; }")
                msg_box.exec_()
            except Exception as e:
                print(f"Failed to save database: {e}")
                self.show_error_message(f"Failed to save database: {e}")

    def add_image_to_db(self):
        if not self.db_path:
            msg_box = QMessageBox()
            msg_box.setStyleSheet("QLabel { color: black; }")
            msg_box.setText("Error")
            msg_box.setInformativeText("No database loaded or created.")
            msg_box.exec_()

            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Add Image to Database", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO images (path) VALUES (?)", (file_path,))
                    conn.commit()
                print(f"Image '{file_path}' added to database.")
                msg_box = QMessageBox()
                msg_box.setText("Image Added")
                msg_box.setInformativeText(f"Image '{file_path}' added to database.")
                msg_box.setStyleSheet("QLabel{ color: black; }")
                msg_box.exec_()
            except Exception as e:
                print(f"Failed to add image to database: {e}")
                self.show_error_message(f"Failed to add image to database: {e}")

    def remove_image_from_db(self):
        if not self.db_path:
            # Show an error message box if no database is loaded
            msg_box = QMessageBox()
            msg_box.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 14px;
                    font-family: 'Arial';
                }
                QMessageBox {
                    background-color: #2C3E50;
                    border-radius: 10px;
                }
                QPushButton {
                    background-color: #1ABC9C;
                    color: white;
                    font-size: 14px;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #16A085;
                }
            """)
            msg_box.setText("Error")
            msg_box.setInformativeText("No database loaded or created.")
            msg_box.exec_()
            return

        try:
            # Fetch the paths from the database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT path FROM images")
                paths = [row[0] for row in cursor.fetchall()]

            if not paths:
                self.show_error_message("No images in the database.")
                return

            # Create a custom dialog to select the image to remove
            dialog = QDialog(self)
            dialog.setWindowTitle("Remove Image from Database")
            dialog.setFixedSize(400, 200)

            layout = QVBoxLayout(dialog)

            # Apply modern styling
            dialog.setStyleSheet("""
                QDialog {
                    background-color: qlineargradient(
                        spread: pad, x1:0, y1:0, x2:1, y2:1,
                        stop:0 #2C3E50, stop:1 #34495E
                    );
                    border-radius: 20px;
                    padding: 20px;
                }
                QLabel {
                    color: white;
                    font-size: 16px;
                    font-family: 'Arial';
                }
                QComboBox {
                    background-color: #34495E;
                    color: white;
                    font-size: 16px;
                    padding: 5px;
                    border: 1px solid white;
                    border-radius: 5px;
                }
                QPushButton {
                    background-color: #1ABC9C;
                    color: white;
                    font-size: 16px;
                    font-family: 'Arial';
                    border: none;
                    padding: 10px 20px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #16A085;
                }
            """)

            # Add a label and combo box for image selection
            label = QLabel("Select an image to remove:")
            layout.addWidget(label)

            combo_box = QComboBox(dialog)
            combo_box.addItems(paths)
            layout.addWidget(combo_box)

            # Create OK and Cancel buttons
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(buttons)

            # Handle button clicks
            def onButtonClick(button):
                if button == QDialogButtonBox.Ok:
                    selected_path = combo_box.currentText()
                    if selected_path:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM images WHERE path=?", (selected_path,))
                            conn.commit()
                        print(f"Image '{selected_path}' removed from database.")
                        # Show success message
                        msg_box = QMessageBox()
                        msg_box.setText("Image Removed")
                        msg_box.setInformativeText(f"Image '{selected_path}' removed from database.")
                        msg_box.setStyleSheet("""
                            QLabel {
                                color: white;
                                font-size: 14px;
                                font-family: 'Arial';
                            }
                            QMessageBox {
                                background-color: #2C3E50;
                                border-radius: 10px;
                            }
                            QPushButton {
                                background-color: #1ABC9C;
                                color: white;
                                font-size: 14px;
                                border: none;
                                padding: 10px 15px;
                                border-radius: 5px;
                            }
                            QPushButton:hover {
                                background-color: #16A085;
                            }
                        """)
                        msg_box.exec_()
                        dialog.accept()
                    else:
                        QMessageBox.warning(self, "Selection Error", "Please select a valid image.")
                elif button == QDialogButtonBox.Cancel:
                    dialog.reject()

            buttons.accepted.connect(lambda: onButtonClick(QDialogButtonBox.Ok))
            buttons.rejected.connect(lambda: onButtonClick(QDialogButtonBox.Cancel))

            dialog.setLayout(layout)
            dialog.exec_()

        except Exception as e:
            print(f"Failed to remove image from database: {e}")
            self.show_error_message(f"Failed to remove image from database: {e}")

    def view_project_description(self):
        description_text = """
            <h2 style="color: white; font-family: 'Arial'; font-size: 28px; text-align: center;">Project Description</h2>
            <p style="color: white; font-family: 'Arial'; font-size: 18px; text-align: justify;">
                This project aims to estimate smoking-associated damage based on nuclear lung images using a trained 
                convolutional neural network (CNN) model. The application allows users to load lung images, perform analysis, 
                and view the results, helping to identify potential damage caused by smoking.
            </p>
            <h3 style="color: white; font-family: 'Arial'; font-size: 22px; text-align: center;">Main Features</h3>
            <ul style="color: white; font-family: 'Arial'; font-size: 16px; text-align: left;">
                <li>Load and analyze lung X-ray images</li>
                <li>Apply advanced deep learning techniques for analysis</li>
                <li>View results with confidence scores</li>
                <li>Visualize annotated lung borders and detected edges</li>
            </ul>
            <p style="color: white; font-family: 'Arial'; font-size: 18px; text-align: justify;">
                The tool helps doctors and researchers to estimate lung damage and make informed decisions 
                regarding lung health, particularly in individuals with a history of smoking.
            </p>
        """

        dialog = QDialog(self)
        dialog.setWindowTitle("Project Description")
        dialog.setFixedSize(600, 500)  # Adjust the size if necessary
        layout = QVBoxLayout(dialog)

        # Apply a modern stylesheet
        dialog.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E
                );
                border-radius: 20px;
                padding: 20px;
            }
            QLabel {
                color: white;
                font-size: 18px;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)

        # Add a shadow effect for depth
        shadow = QGraphicsDropShadowEffect(dialog)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 150))
        dialog.setGraphicsEffect(shadow)

        label = QLabel(description_text, dialog)
        label.setTextFormat(Qt.RichText)  # Use RichText to apply HTML styling
        label.setWordWrap(True)  # Enable word wrapping for long text
        layout.addWidget(label)

        # Add a Close button with modern styling
        close_button = QPushButton("Close", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec_()

    def view_analysis_options(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Options of Analysis")
        dialog.setFixedSize(400, 600)  # Adjust size for more options
        layout = QVBoxLayout(dialog)
        dialog.setStyleSheet("""
                QLabel, QCheckBox, QSlider {
                    color: white;
                    font-size: 20px;
                    font-family: 'Arial';
                }
                QDialog {
                    background-color: #34495E;
                    border-radius: 15px;
                    padding: 20px;
                }
                QPushButton {
                    background-color: #1ABC9C;
                    color: white;
                    font-size: 20px;
                    font-family: 'Arial';
                    border: none;
                    padding: 10px 20px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #16A085;
                }
            """)
        # Existing options
        show_edges_checkbox = QCheckBox("Show Edges")
        show_edges_checkbox.setChecked(self.analysis_options['show_edges'])
        layout.addWidget(show_edges_checkbox)

        annotate_borders_checkbox = QCheckBox("Annotate Borders")
        annotate_borders_checkbox.setChecked(self.analysis_options['annotate_borders'])
        layout.addWidget(annotate_borders_checkbox)

        # New options
        noise_reduction_checkbox = QCheckBox("Apply Noise Reduction")
        noise_reduction_checkbox.setChecked(self.analysis_options.get('noise_reduction', False))
        layout.addWidget(noise_reduction_checkbox)

        edge_sensitivity_slider = QSlider(Qt.Horizontal)
        edge_sensitivity_slider.setMinimum(1)
        edge_sensitivity_slider.setMaximum(10)
        edge_sensitivity_slider.setValue(self.analysis_options.get('edge_sensitivity', 5))
        edge_sensitivity_slider.setTickInterval(1)
        edge_sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Edge Detection Sensitivity"))
        layout.addWidget(edge_sensitivity_slider)

        roi_selection_checkbox = QCheckBox("Select Region of Interest (ROI)")
        roi_selection_checkbox.setChecked(self.analysis_options.get('roi_selection', False))
        layout.addWidget(roi_selection_checkbox)

        image_scaling_slider = QSlider(Qt.Horizontal)
        image_scaling_slider.setMinimum(50)
        image_scaling_slider.setMaximum(200)
        image_scaling_slider.setValue(self.analysis_options.get('image_scaling', 100))
        image_scaling_slider.setTickInterval(10)
        image_scaling_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Image Scaling (%)"))
        layout.addWidget(image_scaling_slider)

        model_selection_combo = QComboBox()
        model_selection_combo.addItems(["Model A", "Model B", "Model C"])
        model_selection_combo.setCurrentText(self.analysis_options.get('model_selection', "Model A"))
        layout.addWidget(QLabel("Model Selection"))
        layout.addWidget(model_selection_combo)

        # Confidence Threshold Slider
        confidence_threshold_slider = QSlider(Qt.Horizontal)
        confidence_threshold_slider.setMinimum(50)
        confidence_threshold_slider.setMaximum(100)
        confidence_threshold_slider.setValue(self.analysis_options.get('confidence_threshold', 75))
        confidence_threshold_slider.setTickInterval(5)
        confidence_threshold_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Confidence Threshold (%)"))
        layout.addWidget(confidence_threshold_slider)

        # Brightness and Contrast sliders
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setMinimum(-100)
        brightness_slider.setMaximum(100)
        brightness_slider.setValue(self.analysis_options.get('brightness', 0))
        brightness_slider.setTickInterval(10)
        brightness_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Brightness"))
        layout.addWidget(brightness_slider)
        ##

        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setMinimum(-100)
        contrast_slider.setMaximum(100)
        contrast_slider.setValue(self.analysis_options.get('contrast', 0))
        contrast_slider.setTickInterval(10)
        contrast_slider.setTickPosition(QSlider.TicksBelow)
        layout.addWidget(QLabel("Contrast"))
        layout.addWidget(contrast_slider)

        save_button = QPushButton("Save", dialog)
        save_button.clicked.connect(lambda: self.save_analysis_options(dialog,
                                                                       show_edges_checkbox.isChecked(),
                                                                       annotate_borders_checkbox.isChecked(),
                                                                       noise_reduction_checkbox.isChecked(),
                                                                       edge_sensitivity_slider.value(),
                                                                       roi_selection_checkbox.isChecked(),
                                                                       image_scaling_slider.value(),
                                                                       model_selection_combo.currentText(),
                                                                       confidence_threshold_slider.value(),
                                                                       brightness_slider.value(),
                                                                       contrast_slider.value()))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_analysis_options(self, dialog, show_edges, annotate_borders, noise_reduction, edge_sensitivity,
                              roi_selection, image_scaling, model_selection, confidence_threshold, brightness,
                              contrast):
        # Save the analysis options
        self.analysis_options['show_edges'] = show_edges
        self.analysis_options['annotate_borders'] = annotate_borders
        self.analysis_options['noise_reduction'] = noise_reduction
        self.analysis_options['edge_sensitivity'] = edge_sensitivity
        self.analysis_options['roi_selection'] = roi_selection
        self.analysis_options['image_scaling'] = image_scaling
        self.analysis_options['model_selection'] = model_selection
        self.analysis_options['confidence_threshold'] = confidence_threshold
        self.analysis_options['brightness'] = brightness
        self.analysis_options['contrast'] = contrast

        # Close the dialog after saving
        dialog.accept()

        # Show a message box confirming the options were saved
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Saved")
        msg_box.setText("Analysis options have been saved successfully.")

        # Apply modern styling to the message box
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #34495E;
                border-radius: 15px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)

        msg_box.exec_()

    def quality_control_action(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File for Quality Control", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            print(f"Selected file: {file_path}")
            analysis_successful = self.quality_control(file_path)
            if analysis_successful:
                print("Analysis successful, displaying results.")
                professional_result = self.fetch_professional_results(file_path)
                manual_result = self.get_manual_input()
                automatic_result = self.analyze_image(file_path)

                combined_results = {
                    'professional': professional_result,
                    'manual': manual_result,
                    'automatic': automatic_result
                }

                if combined_results:
                    self.show_quality_control_results(file_path, combined_results)
                else:
                    print("Quality control or analysis failed.")
                    self.show_error_message("Quality control failed or image analysis failed.")
            else:
                print("Quality control or analysis failed.")
                self.show_error_message("Quality control failed or image analysis failed.")

    def view_data_format(self):
        print("view_data_format triggered")  # Debugging print

        # Example database options
        database_options = ["Data input 1", "Data input 2", "Data input 3"]  # Example options

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Data Input/Output Format")
        dialog.setFixedSize(400, 200)  # Adjust the size if necessary

        layout = QVBoxLayout(dialog)

        # Apply a modern stylesheet
        dialog.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(
                    spread: pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2C3E50, stop:1 #34495E
                );
                border-radius: 20px;
                padding: 20px;
            }
            QLabel, QComboBox {
                color: white;
                font-size: 16px;
                font-family: 'Arial';
            }
            QComboBox {
                background-color: #34495E;
                padding: 5px;
                border: 1px solid white;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)

        # Create a combo box to display database options
        combo_box = QComboBox()
        combo_box.addItems(database_options)
        layout.addWidget(combo_box)

        # Create buttons for selecting and canceling with modern design
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # Add styles to the buttons
        buttons.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
            QPushButton {
                background-color: #1ABC9C;
                color: white;
                font-size: 16px;
                font-family: 'Arial';
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #16A085;
            }
        """)

        layout.addWidget(buttons)

        # Handle button clicks
        def onButtonClick(button):
            if button == QDialogButtonBox.Ok:
                selected_database = combo_box.currentText()
                print(f"Selected: {selected_database}")  # Debugging print
                self.analyze_and_show_histogram(selected_database)
                dialog.accept()
            elif button == QDialogButtonBox.Cancel:
                dialog.reject()

        buttons.accepted.connect(lambda: onButtonClick(QDialogButtonBox.Ok))
        buttons.rejected.connect(lambda: onButtonClick(QDialogButtonBox.Cancel))

        dialog.setLayout(layout)
        dialog.exec_()

    def load_database(self, selected_database):
        try:
            # Determine the path based on selected_database
            if selected_database == "Data input 1":
                database_path = os.path.join(os.getcwd(), "datainput1.sql")
            elif selected_database == "Data input 2":
                database_path = os.path.join(os.getcwd(), "datainput2.sql")
            elif selected_database == "Data input 3":
                database_path = os.path.join(os.getcwd(), "datainput3.sql")

            # Example loading process (replace with your actual loading code)
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            # Perform lung data analysis and plot histogram
            self.analyze_and_plot_histogram(selected_database)

            # Example query
            cursor.execute("SELECT * FROM images")
            rows = cursor.fetchall()

            # Process the loaded data as needed
            for row in rows:
                print(row)

            # Close connection
            conn.close()

        except sqlite3.Error as e:
            # Handle SQLite errors
            print("SQLite error:", e)
            self.show_message("Error Loading Database", f"SQLite error: {str(e)}", QMessageBox.Critical)

        except Exception as e:
            # Handle other exceptions
            print("Error:", e)
            self.show_message("Error Loading Database", f"Error: {str(e)}", QMessageBox.Critical)

    def show_message(self, title, message, icon):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec_()

    def analyze_and_show_histogram(self, selected_input):
        if selected_input == "Data input 1":
            self.show_histogram_input_1()
        elif selected_input == "Data input 2":
            self.show_histogram_input_2()
        elif selected_input == "Data input 3":
            self.show_histogram_input_3()
        else:
            print("Invalid input selected.")

    def plot_histogram(self, normal_count, pneumonia_count):
        print(f"Plotting histogram with Normal={normal_count}, Pneumonia={pneumonia_count}")  # Debugging print
        labels = ['Normal', 'Pneumonia']
        counts = [normal_count, pneumonia_count]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color=['#4CAF50', '#F44336'], edgecolor='black', linewidth=1.2)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom')

        plt.xlabel('Lung Condition', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=14, fontweight='bold')
        plt.title('Histogram of Lung Conditions', fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        print("Displaying the plot")  # Debugging print
        plt.show(block=False)

        QApplication.processEvents()

    def view_results(self):
        if not self.db_path:
            msg_box = QMessageBox()
            msg_box.setStyleSheet("QLabel { color: black; }")
            msg_box.setText("Error")
            msg_box.setInformativeText("No database loaded or created.")
            msg_box.exec_()
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM images")
                images = cursor.fetchall()

                if not images:
                    self.show_error_message("No images found in the database.")
                    return

                # Display all the images and their analysis results
                for image_id, image_path in images:
                    analysis_result = self.analyze_image(image_path)
                    if not analysis_result:
                        continue

                    probability = analysis_result['probability']
                    disease = analysis_result['disease']
                    result_text = f"Image ID: {image_id}, Probability: {probability:.2f}, Disease: {disease}"
                    self.show_info_message("Analysis Result", result_text)
        except Exception as e:
            print(f"Failed to fetch results from database: {e}")
            self.show_error_message(f"Failed to fetch results from database: {e}")

    def show_info_message(self, title, text):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #f5f5f5;
                border-radius: 15px;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
        """)
        msg_box.exec_()

    def show_error_message(self, message):
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle('Error')
        error_box.setText(message)
        print(message)
        error_box.setStyleSheet("QLabel { color : #ffffff; } QMessageBox { background-color: #2b2b2b; }")

        error_box.exec_()


class AnalysisResultDialog(QDialog):
    def __init__(self, original_image_path, labeled_image, edges_image, lung_borders_image, noise_reduced_image,
                 analysis_results, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Analysis Result')
        self.setFixedSize(1000, 900)  # Adjust the size as needed

        layout = QVBoxLayout()

        # Add shadow effect to the dialog
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.setGraphicsEffect(shadow)

        # Image scaling factor
        scale_factor = 300

        # Modern and attractive font styling
        font_style = "font-family: 'Arial'; font-size: 18px; font-weight: bold; color: white;"

        # Display Original Image with label
        original_label = QLabel(self)
        original_image = QPixmap(original_image_path)
        original_label.setPixmap(original_image.scaled(scale_factor, scale_factor, aspectRatioMode=1))
        original_label.setAlignment(Qt.AlignCenter)

        original_text = QLabel("Original Lungs Image", self)
        original_text.setAlignment(Qt.AlignCenter)
        original_text.setStyleSheet(font_style)

        # Display Labeled Image with label
        labeled_label = QLabel(self)
        labeled_image_qt = QImage(labeled_image.data, labeled_image.shape[1], labeled_image.shape[0],
                                  labeled_image.strides[0], QImage.Format_RGB888)
        labeled_pixmap = QPixmap.fromImage(labeled_image_qt)
        labeled_label.setPixmap(labeled_pixmap.scaled(scale_factor, scale_factor, aspectRatioMode=1))
        labeled_label.setAlignment(Qt.AlignCenter)

        labeled_text = QLabel("Labeled Image with Detections", self)
        labeled_text.setAlignment(Qt.AlignCenter)
        labeled_text.setStyleSheet(font_style)

        images_layout = QGridLayout()
        images_layout.addWidget(original_label, 0, 0)
        images_layout.addWidget(original_text, 1, 0)
        images_layout.addWidget(labeled_label, 0, 1)
        images_layout.addWidget(labeled_text, 1, 1)

        if edges_image is not None:
            # Display Edge Image with label
            edges_label = QLabel(self)
            edges_image_qt = QImage(edges_image.data, edges_image.shape[1], edges_image.shape[0],
                                    edges_image.strides[0], QImage.Format_Grayscale8)
            edges_pixmap = QPixmap.fromImage(edges_image_qt)
            edges_label.setPixmap(edges_pixmap.scaled(scale_factor, scale_factor, aspectRatioMode=1))
            edges_label.setAlignment(Qt.AlignCenter)

            edges_text = QLabel("Edges Detected Image", self)
            edges_text.setAlignment(Qt.AlignCenter)
            edges_text.setStyleSheet(font_style)

            images_layout.addWidget(edges_label, 2, 0)
            images_layout.addWidget(edges_text, 3, 0)

        if lung_borders_image is not None:
            # Display Lung Borders Image with label
            lung_borders_label = QLabel(self)
            lung_borders_image_qt = QImage(lung_borders_image.data, lung_borders_image.shape[1],
                                           lung_borders_image.shape[0], lung_borders_image.strides[0],
                                           QImage.Format_RGB888)
            lung_borders_pixmap = QPixmap.fromImage(lung_borders_image_qt)
            lung_borders_label.setPixmap(lung_borders_pixmap.scaled(scale_factor, scale_factor, aspectRatioMode=1))
            lung_borders_label.setAlignment(Qt.AlignCenter)

            lung_borders_text = QLabel("Lung Borders Detected Image", self)
            lung_borders_text.setAlignment(Qt.AlignCenter)
            lung_borders_text.setStyleSheet(font_style)

            images_layout.addWidget(lung_borders_label, 2, 1)
            images_layout.addWidget(lung_borders_text, 3, 1)

        if noise_reduced_image is not None:
            # Display Noise Reduced Image with label
            noise_reduced_label = QLabel(self)
            noise_reduced_image_qt = QImage(noise_reduced_image.data, noise_reduced_image.shape[1],
                                            noise_reduced_image.shape[0], noise_reduced_image.strides[0],
                                            QImage.Format_RGB888)
            noise_reduced_pixmap = QPixmap.fromImage(noise_reduced_image_qt)
            noise_reduced_label.setPixmap(noise_reduced_pixmap.scaled(scale_factor, scale_factor, aspectRatioMode=1))
            noise_reduced_label.setAlignment(Qt.AlignCenter)

            noise_reduced_text = QLabel("Noise Reduced Image", self)
            noise_reduced_text.setAlignment(Qt.AlignCenter)
            noise_reduced_text.setStyleSheet(font_style)

            images_layout.addWidget(noise_reduced_label, 4, 0)
            images_layout.addWidget(noise_reduced_text, 5, 0)

        layout.addLayout(images_layout)

        # Display Analysis Results
        result_text = QLabel(self)
        result_text.setText(f"Analysis Results:\n")
        for result in analysis_results:
            result_text.setText(
                result_text.text() + f"{result['disease']} with confidence {result['confidence']:.2f}\n")
        result_text.setAlignment(Qt.AlignCenter)
        result_text.setStyleSheet(font_style)

        layout.addWidget(result_text)

        self.setLayout(layout)

        # Add a Close button
        close_button = QPushButton('Close', self)
        close_button.setStyleSheet(
            "background-color: #007BFF; color: white; font-size: 16px; padding: 10px 20px; border-radius: 5px;")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)


import tempfile


class ContrastAdjustmentDialog(QDialog):
    def __init__(self, parent=None, image=None, update_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Contrast Adjustment")
        self.layout = QVBoxLayout(self)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.layout.addWidget(QLabel("Brightness"))
        self.layout.addWidget(self.brightness_slider)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.layout.addWidget(QLabel("Contrast"))
        self.layout.addWidget(self.contrast_slider)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_contrast)
        self.layout.addWidget(self.apply_button)

        self.setLayout(self.layout)

        self.image = image
        self.update_callback = update_callback
        self.enhanced_image = None
        self.enhanced_image_path = None

    def apply_contrast(self):
        if self.image is None:
            QMessageBox.warning(self, "Error", "No image loaded!")
            return

        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        try:
            # Convert to float to prevent clipping values during transformation
            img_float = self.image.astype(np.float32)

            # Apply contrast and brightness
            alpha = contrast / 100.0 + 1
            beta = brightness
            img_adjusted = cv2.convertScaleAbs(img_float, alpha=alpha, beta=beta)

            # Save the enhanced image to a temporary file
            _, temp_filename = tempfile.mkstemp(suffix=".png")
            cv2.imwrite(temp_filename, cv2.cvtColor(img_adjusted, cv2.COLOR_RGB2BGR))

            self.enhanced_image = img_adjusted
            self.enhanced_image_path = temp_filename
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply contrast: {e}")

    def get_enhanced_image(self):
        return self.enhanced_image

    def get_enhanced_image_path(self):
        return self.enhanced_image_path


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setFixedSize(400, 300)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QLabel {
                font-size: 16px;
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.layout = QVBoxLayout(self)

        self.message_label = QLabel(self)
        self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout.addWidget(self.message_label)

        self.close_button = QPushButton("OK", self)
        self.close_button.clicked.connect(self.accept)
        self.layout.addWidget(self.close_button)

    def add_message(self, message):
        self.message_label.setText(message)

    def clear_messages(self):
        self.message_label.clear()


if __name__ == '__main__':
    app = QApplication([])
    window = MedicalApp()
    window.show()
    app.exec_()