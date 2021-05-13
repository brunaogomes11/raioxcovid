# Interface Gráfica
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication, QWidget, QGridLayout, QFileDialog, QMessageBox, QAction
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon

class MyWindow(QMainWindow):
    
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setup_main_window()
        self.initUI()

    def setup_main_window(self):
        self.x = 1000
        self.y = 650
        self.setMinimumSize(QSize(self.x, self.y))
        self.setWindowTitle("Raio-X-Covid")
        self.wid = QWidget(self)
        self.setCentralWidget(self.wid)
        self.layout = QGridLayout()
        self.wid.setLayout(self.layout)
        self.wid.setStyleSheet("""
            QWidget {
                background: #EFEFEF;
            }
            QPushButton {
                background: #505050;
                border-radius: 5px;
                font-size: 14px;
                font-weight: 600;
                color: white;
                min-height: 40px;
                min-width: 150px;
            }
            QPushButton:hover {
                background: #000000;
            }
            QLabel {
                color: black;
                font-size: 18px;
            }
            QLabel#imagemEscolhida {
                border: 1px solid #DADADA;
                border-radius: 10px;
            }
            QMessageBox {
                color: blue;
            }
            #msgAviso {
                font-size: 12px;
                font-weight: bold;
            }
        """)

    def initUI(self):
        #Pop-Up inicial
        self.showWarning()
        #Barra de Menu
        self.barraMenu = self.menuBar()
        self.files = self.barraMenu.addMenu("&Arquivo")
        self.abrirImagem = self.files.addAction("Abrir Imagem")
        self.abrirImagem.triggered.connect(self.getFile)
        self.abrirImagem.setShortcut("Ctrl+O")
        self.fecharImagem = self.files.addAction("Fechar Imagem")
        self.fecharImagem.triggered.connect(self.closeFile)
        self.fecharImagem.setShortcut("Ctrl+F")
        self.files.addSeparator()
        self.iniciarAnalise = self.files.addAction("Fazer Análise")
        self.iniciarAnalise.triggered.connect(self.classificar_imagem)
        self.iniciarAnalise.setShortcut("Ctrl+A")
        self.files.addSeparator()
        self.fecharProgram = self.files.addAction("Fechar Programa")
        self.fecharProgram.triggered.connect(self.exitProgram)
        self.fecharProgram.setShortcut("Alt+F4")
        self.sobre = self.barraMenu.addMenu("&Sobre")
        self.ajuda = self.sobre.addAction("Ajuda")
        self.ajuda.triggered.connect(self.ajudaPainel)
        self.ajuda.setShortcut("Ctrl+H")
        self.sobre.addSeparator()
        self.sobreApp = self.sobre.addAction("Sobre o Aplicativo")
        self.sobreApp.setShortcut("Ctrl+G")
        self.sobreApp.triggered.connect(self.sobreAppFuc)
        self.sobreDev = self.sobre.addAction("Sobre o Desenvolvedor")
        self.sobreDev.setShortcut("Ctrl+D")
        self.sobreDev.triggered.connect(self.sobreDevFuc)

        #Barra de Status
        self.barraStatus = self.statusBar()
        self.barraStatus.showMessage("Parado")

        #Barra de Icones
        self.iconToolbar = self.addToolBar("Arquivo")
        self.iconToolbar.setMovable(False)
        
        self.openIcon = self.iconToolbar.addAction(QIcon("./icons/openIcon.png"), "Abrir Imagem")
        self.openIcon.triggered.connect(self.getFile)
        self.closeIcon = self.iconToolbar.addAction(QIcon("./icons/closeIcon.png"), "Fechar Imagem")
        self.closeIcon.triggered.connect(self.closeFile)
        self.testIcon = self.iconToolbar.addAction(QIcon("./icons/testIcon.png"), "Fazer Análise")
        self.testIcon.triggered.connect(self.classificar_imagem)
        self.helpIcon = self.iconToolbar.addAction(QIcon("./icons/helpIcon.png"), "Abrir Painel de Ajuda")
        self.helpIcon.triggered.connect(self.ajudaPainel)
        
        #Botoes
        self.buttonAddFile = QtWidgets.QPushButton(self)
        self.buttonAddFile.setText("Abrir Imagem")
        self.buttonAddFile.clicked.connect(self.getFile)
        self.buttonTestar = QtWidgets.QPushButton(self)
        self.buttonTestar.setText("Fazer Análise")
        self.buttonTestar.clicked.connect(self.classificar_imagem)
        
        self.p = None
        self.fname = None

        self.resultadoText = QLabel("", self)
        self.resultadoText.setAlignment(Qt.AlignCenter)
        self.msgAviso = QLabel("", self)
        self.msgAviso.setObjectName("msgAviso")
        self.msgAviso.setWordWrap(True)
        #Imagens
        self.imagemEscolhida = QLabel("", self)
        self.imagemEscolhida.setObjectName('imagemEscolhida')

        self.layout.addWidget(self.buttonTestar, 1, 2)
        self.layout.addWidget(self.buttonAddFile, 0, 2)
        self.layout.addWidget(self.imagemEscolhida, 0, 0, 4, 2)
        self.layout.addWidget(self.resultadoText, 2, 2)
        self.layout.addWidget(self.msgAviso, 3, 2)
        
        
    def showWarning(self):
        self.msgWarning = QMessageBox()
        self.msgWarning.setWindowTitle("Aviso Importante")
        self.msgWarning.setIcon(QMessageBox.Warning)
        self.msgText = "Esse aplicativo NÃO é a solução definitiva para identificação de Covid ou Pneumonia. Testes realizados com o aplicativo apresentam uma precisão de apenas 79% na classificação das imagens.\n\nÉ importante alertar que esse aplicativo NÃO é 100% confiável. NÃO pode ser usado sem acompanhamento médico. NÃO pode ser usado como única forma de diagnóstico. NÃO pode ser usado por pacientes."
        self.msgWarning.setText(self.msgText)
        self.msgWarning.exec()

    def exitProgram(self):
        sys.exit()

    def ajudaPainel(self):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Ajuda")
        self.msg.setText("Raio-X-Covid\nVersão 1.0")
        self.ajudaProg = "Como utilizar esse software:\nAbra uma imagem de um raio-x de pulmão. Clique em Fazer Análise. Espere alguns segundos e o programa classificará, com certo grau de precisão, a imagem como: COVID, Pneumonia ou nenhum destes.\n"
        self.atalhosTexto = f"{self.ajudaProg}\nAtalhos\nCtrl+O: Abrir Imagem para classificação\nCtrl+F: Fechar Imagem\nCtrl+A: Fazer Análise\n\nCtrl+H: Painel de ajuda\nCtrl+G: Painel sobre o aplicativo\nCtrl+D: Painel sobre o desenvolvedor\nAlt+F4: Fechar Programa"
        self.msg.setInformativeText(self.atalhosTexto)
        self.msg.exec_()

    def sobreAppFuc(self):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Sobre o Aplicativo")
        self.msg.setText("O aplicativo Raio-X-Covid foi desenvolvido como fruto de um projeto de pesquisa do Instituto Federal do Triângulo Mineiro campus Ituiutaba. Seu objetivo é auxiliar a medicina diagnóstica na identificação de imagens de raio-x pulmonares com suspeita de Covid-19 ou Pneumonia.\nEsse aplicativo NÃO substitui o diagnóstico médico.\nEsse aplicativo NÃO deve ser utilizado como única forma de diagnóstico médico por imagens.\nEsse aplicativo NÃO deve ser utilizado por pacientes sem o acompanhamento médico.")
        self.msg.exec_()

    def sobreDevFuc(self):
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Sobre O Desenvolvedor")
        self.textoDev = "Desenvolvido por Bruno Gomes Pereira, sob orientação do professor André Luiz França Batista, como fruto de um projeto de pesquisa no Instituto Federal do Triângulo Mineiro."
        self.msg.setText(self.textoDev)
        self.msg.exec_()

    def getFile(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Selecionar Arquivo', '', "Image files (*.jpg *.png *.jpeg *.bmp *.webp *.tiff)")
        if (self.fname):
            self.imagemEscolhida.setPixmap(QPixmap(self.fname).scaled(650, 650))
            self.imagemEscolhida.setAlignment(Qt.AlignCenter)
            self.resultadoText.setText("")

    def classificar_imagem(self):
        if (self.fname):
            import torch
            from torch import nn, optim
            import torch.nn.functional as F
            from torchvision import datasets, transforms
            torch.manual_seed(123)
            # Contrução do Modelo
            class classificador(nn.Module):
                def __init__(self):
                    super().__init__()

                    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
                    self.conv2 = nn.Conv2d(64, 64, (3,3))
                    self.activation = nn.ReLU()
                    self.bnorm = nn.BatchNorm2d(num_features=64)
                    self.pool = nn.MaxPool2d(kernel_size=(2,2))
                    self.flatten = nn.Flatten()

                    # output = (input - filter + 1) / stride
                    # Convolução 1 -> (64 - 3 + 1) / 1 = 62x62
                    # Pooling 1 -> Só dividir pelo kernel_size = 31x31
                    # Convolução 2 -> (31 - 3 + 1)/ 1 = 29x29
                    # Pooling 2 -> Só dividir pelo kernel_size = 14x14
                    # 14 * 14 * 64
                    # 33907200 valores -> 256 neurônios da camada oculta
                    self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
                    self.linear2 = nn.Linear(256, 128)
                    self.output = nn.Linear(128, 3)

                def forward(self, X):
                    X = self.pool(self.bnorm(self.activation(self.conv1(X))))
                    X = self.pool(self.bnorm(self.activation(self.conv2(X))))
                    X = self.flatten(X)

                    # Camadas densas
                    X = self.activation(self.linear1(X))
                    X = self.activation(self.linear2(X))
                    
                    # Saída
                    X = self.output(X)

                    return X

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device

            classificadorLoaded = classificador()
            state_dict = torch.load('checkpoint.pth')
            classificadorLoaded.load_state_dict(state_dict)
            self.resultadoText.setText(f"Classificando...")
            self.barraStatus.showMessage("Classificando Imagem...")
            from PIL import Image
            self.imagem_teste = Image.open(self.fname)

            import numpy as np
            self.imagem_teste = self.imagem_teste.resize((64, 64))
            self.imagem_teste = self.imagem_teste.convert('RGB') 
            self.imagem_teste = np.array(self.imagem_teste.getdata()).reshape(*self.imagem_teste.size, -1)
            self.imagem_teste = self.imagem_teste / 255
            self.imagem_teste = self.imagem_teste.transpose(2, 0, 1)
            self.imagem_teste = torch.tensor(self.imagem_teste, dtype=torch.float).view(-1, *self.imagem_teste.shape)
            
            classificadorLoaded.eval()
            self.imagem_teste = self.imagem_teste.to(device)
            self.output = classificadorLoaded.forward(self.imagem_teste)
            self.output = F.softmax(self.output, dim=1)
            self.top_p, self.top_class = self.output.topk(k = 1, dim = 1)
            self.output = self.output.detach().numpy()
            self.resultado = np.argmax(self.output)

            if (self.resultado == 1):
                self.resultadoText.setText("Imagem identificada como:\nPNEUMONIA não detectada\nCOVID não detectado")
            elif (self.resultado == 2) :
                self.resultadoText.setText("Imagem identificada como:\nPNEUMONIA")
            else:
                self.resultadoText.setText("Imagem identificada como:\nCOVID")
            self.barraStatus.showMessage("Imagem Classificada", 5000)
            self.msgAviso.setText("Esse aplicativo NÃO é a solução definitiva para identificação de Covid ou Pneumonia. Testes realizados com o aplicativo apresentaram uma precisão de apenas 75% na classificação das imagens.\n\nÉ importante alertar que esse aplicativo NÃO é 100% confiável. NÃO pode ser usado sem acompanhamento médico. NÃO pode ser usado como única forma de diagnóstico. NÃO pode ser usado por pacientes.")
        else:
            self.msg = QMessageBox()
            self.msg.setWindowTitle("Erro")
            self.msg.setIcon(QMessageBox.Critical)
            self.msg.setText("Você precisa selecionar uma imagem")
            self.msg.exec_()

    def closeFile(self):
        if (self.fname):
            self.imagemEscolhida.setText(" ")
            self.resultadoText.setText(" ")
            self.fname = None

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()
