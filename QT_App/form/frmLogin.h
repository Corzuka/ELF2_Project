#ifndef FRMLOGIN_H
#define FRMLOGIN_H

#include <QDialog>

// 前向声明
namespace Ui {
class frmLogin;
}

class frmLogin : public QDialog
{
    Q_OBJECT

public:
    explicit frmLogin(QWidget *parent = nullptr);
    ~frmLogin();

    static bool authenticate(const QString &username, const QString &password);

private slots:
    void on_btnLogin_clicked();
    void on_btnCancel_clicked();

private:
    Ui::frmLogin *ui;  // 这里使用不完整类型是可以的
};

#endif // FRMLOGIN_H
