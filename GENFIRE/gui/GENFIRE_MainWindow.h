#ifndef GENFIRE_MainWindow_H
#define GENFIRE_MainWindow_H

#include <QMainWindow>

namespace Ui {
class GENFIRE_MainWindow;
}

class GENFIRE_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit GENFIRE_MainWindow(QWidget *parent = 0);
    ~GENFIRE_MainWindow();

private:
    Ui::GENFIRE_MainWindow *ui;
};

#endif // GENFIRE_MainWindow_H
