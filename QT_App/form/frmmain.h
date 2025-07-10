#ifndef FRMMAIN_H
#define FRMMAIN_H

#include <QWidget>

namespace Ui {
class frmMain;
}

class frmMain : public QWidget
{
    Q_OBJECT

public:
    explicit frmMain(QWidget *parent = nullptr);  // 改为使用nullptr
    ~frmMain();

    void showMainInterface();  // 新增公开方法

private:
    Ui::frmMain *ui;

public slots:
    void initForm();
    void initConfig();
    void saveConfig();
};

#endif // FRMMAIN_H
