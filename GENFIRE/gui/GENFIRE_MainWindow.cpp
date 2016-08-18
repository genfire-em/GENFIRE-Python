#include "GENFIRE_MainWindow.h"
#include "ui_GENFIRE_MainWindow.h"

GENFIRE_MainWindow::GENFIRE_MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GENFIRE_MainWindow)
{
    ui->setupUi(this);
}

GENFIRE_MainWindow::~GENFIRE_MainWindow()
{
    delete ui;
}
