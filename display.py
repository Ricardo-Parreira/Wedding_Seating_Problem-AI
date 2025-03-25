import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, 
    QMessageBox, QStackedWidget, QComboBox, QSpinBox, QFormLayout
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from project_notebook import run_wedding_seating, random_preferences


class StartScreen(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Centraliza os elementos

        title = QLabel("Welcome to Wedding Seating Planner", self)
        title.setFont(QFont("Arial", 18))
        layout.addWidget(title)

        start_button = QPushButton("Start", self)
        start_button.setFont(QFont("Arial", 14))
        start_button.clicked.connect(self.go_to_main_screen)
        start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; border-radius: 5px;")
        layout.addWidget(start_button)

        self.setLayout(layout)

    def go_to_main_screen(self):
        self.stacked_widget.setCurrentIndex(1)  # Muda para a tela principal


class WeddingPlannerApp(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()
        self.preference_matrix = None 

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Wedding Seating Planner", self)
        title.setFont(QFont("Arial", 16))
        layout.addWidget(title)

        # Criando formulário para entrada de dados
        form_layout = QFormLayout()

        # Dropdown para selecionar algoritmo
        self.algorithm_dropdown = QComboBox()
        self.algorithm_dropdown.addItems(["Genetic Algorithm version1", "Genetic Algorithm version2", "Simulated Annealing", "Tabu Search"])
        form_layout.addRow("Escolha o Algoritmo:", self.algorithm_dropdown)

        # Número de convidados
        self.num_guests = QSpinBox()
        self.num_guests.setRange(1, 1000)  # Limite máximo de 1000 convidados
        self.num_guests.valueChanged.connect(self.update_preference_matrix)
        form_layout.addRow("Número de Convidados:", self.num_guests)

        # Número de mesas
        self.num_tables = QSpinBox()
        self.num_tables.setRange(1, 50)
        form_layout.addRow("Número de Mesas:", self.num_tables)

        # Número de assentos por mesa
        self.seats_per_table = QSpinBox()
        self.seats_per_table.setRange(1, 20)
        form_layout.addRow("Assentos por Mesa:", self.seats_per_table)

        layout.addLayout(form_layout)

        # Botão para gerar planejamento
        self.button = QPushButton("Gerar Planejamento", self)
        self.button.setFont(QFont("Arial", 14))
        self.button.clicked.connect(self.gerar_planejamento)
        self.button.setStyleSheet("background-color: #008CBA; color: white; font-size: 16px; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.button)

        # Botão para voltar ao menu inicial
        self.back_button = QPushButton("Voltar", self)
        self.back_button.clicked.connect(self.voltar_tela_inicial)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def update_preference_matrix(self):
        """
        Atualiza a matriz de preferências apenas quando o número de convidados mudar.
        """
        num_guests = self.num_guests.value()
        if self.preference_matrix is None or num_guests != len(self.preference_matrix):
            self.preference_matrix = random_preferences(num_guests)  # Gera a matriz de preferências
            print(f"Matriz de preferências gerada para {num_guests} convidados.")


    def gerar_planejamento(self):
        algorithm = self.algorithm_dropdown.currentText()
        num_guests = self.num_guests.value()
        num_tables = self.num_tables.value()
        seats_per_table = self.seats_per_table.value()



        resultado = run_wedding_seating(num_guests, num_tables, seats_per_table, algorithm, self.preference_matrix)

        # Exibindo o resultado na tela
        QMessageBox.information(self, "Planejamento", f"Resultado do Algoritmo:\n{resultado}")

        

    def voltar_tela_inicial(self):
        self.stacked_widget.setCurrentIndex(0)  # Volta para a tela inicial


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wedding Seating Planner")
        self.setGeometry(100, 100, 800, 600)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.start_screen = StartScreen(self.stacked_widget)
        self.main_screen = WeddingPlannerApp(self.stacked_widget)

        self.stacked_widget.addWidget(self.start_screen)  # Tela inicial
        self.stacked_widget.addWidget(self.main_screen)   # Tela principal

        self.stacked_widget.setCurrentIndex(0)  # Começa na tela inicial


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
