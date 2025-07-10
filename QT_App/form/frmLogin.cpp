#include "frmlogin.h"
#include "ui_frmlogin.h"  // 必须包含这个自动生成的头文件
#include "qthelper.h"

frmLogin::frmLogin(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::frmLogin)
{
    ui->setupUi(this);

    // 初始化UI设置
    this->setWindowTitle("用户登录");
    this->setFixedSize(400, 300);

    // 设置密码框回显模式
    ui->txtPassword->setEchoMode(QLineEdit::Password);

    // 连接信号槽（如果使用自动连接则不需要）
    // connect(ui->btnLogin, &QPushButton::clicked, this, &frmLogin::on_btnLogin_clicked);
}

frmLogin::~frmLogin()
{
    delete ui;
}

bool frmLogin::authenticate(const QString &username, const QString &password)
{
    // 实际项目中应该使用更安全的验证方式
    return username == "admin" && password == "123456";
}

void frmLogin::on_btnLogin_clicked()
{
    QString username = ui->txtUsername->text().trimmed();
    QString password = ui->txtPassword->text();

    if(username.isEmpty()) {
        QtHelper::showMessageBoxError("用户名不能为空!");
        ui->txtUsername->setFocus();
        return;
    }

    if(password.isEmpty()) {
        QtHelper::showMessageBoxError("密码不能为空!");
        ui->txtPassword->setFocus();
        return;
    }

    if(authenticate(username, password)) {
        accept();  // 关闭对话框并返回QDialog::Accepted
    } else {
        QtHelper::showMessageBoxError("用户名或密码错误!");
        ui->txtPassword->clear();
        ui->txtPassword->setFocus();
    }
}

void frmLogin::on_btnCancel_clicked()
{
    reject();  // 关闭对话框并返回QDialog::Rejected
}
