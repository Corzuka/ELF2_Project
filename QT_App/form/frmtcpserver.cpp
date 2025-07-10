#include "frmtcpserver.h"
#include "ui_frmtcpserver.h"
#include "qthelper.h"
#include "databasemanager.h"

frmTcpServer::frmTcpServer(QWidget *parent) : QWidget(parent), ui(new Ui::frmTcpServer)
{
    ui->setupUi(this);

    // 初始化数据库
    dbManager = new DatabaseManager(this);
    if (!dbManager->initDatabase()) {
        append(3, "数据库初始化失败!");
    }
    this->initForm();
    this->initConfig();
}

frmTcpServer::~frmTcpServer()
{
    //结束的时候停止服务
    server->stop();
    delete ui;
}

bool frmTcpServer::eventFilter(QObject *watched, QEvent *event)
{
    //双击清空
    if (watched == ui->txtMain->viewport()) {
        if (event->type() == QEvent::MouseButtonDblClick) {
            on_btnClear_clicked();
        }
    }

    return QWidget::eventFilter(watched, event);
}

void frmTcpServer::initForm()
{
    QFont font;
    font.setPixelSize(16);
    ui->txtMain->setFont(font);
    ui->txtMain->viewport()->installEventFilter(this);

    isOk = false;

    //实例化对象并绑定信号槽
    server = new TcpServer(this);
    connect(server, SIGNAL(connected(QString, int)), this, SLOT(connected(QString, int)));
    connect(server, SIGNAL(disconnected(QString, int)), this, SLOT(disconnected(QString, int)));
    connect(server, SIGNAL(error(QString, int, QString)), this, SLOT(error(QString, int, QString)));
    connect(server, SIGNAL(sendData(QString, int, QString)), this, SLOT(sendData(QString, int, QString)));
    connect(server, SIGNAL(receiveData(QString, int, QString)), this, SLOT(receiveData(QString, int, QString)));

    //定时器发送数据
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(on_btnSend_clicked()));

    //填充数据到下拉框
    ui->cboxInterval->addItems(AppData::Intervals);
    ui->cboxData->addItems(AppData::Datas);
    ui->cboxListenIP->addItems(QtHelper::getLocalIPs()); // 添加本地IP列表
    ui->cboxListenIP->setCurrentText(AppConfig::TcpListenIP); // 设置当前IP
}

void frmTcpServer::initConfig()
{
    ui->ckHexSend->setChecked(AppConfig::HexSendTcpServer);
    connect(ui->ckHexSend, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    ui->ckHexReceive->setChecked(AppConfig::HexReceiveTcpServer);
    connect(ui->ckHexReceive, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    ui->ckAscii->setChecked(AppConfig::AsciiTcpServer);
    connect(ui->ckAscii, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    ui->ckDebug->setChecked(AppConfig::DebugTcpServer);
    connect(ui->ckDebug, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    ui->ckAutoSend->setChecked(AppConfig::AutoSendTcpServer);
    connect(ui->ckAutoSend, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    ui->cboxInterval->setCurrentIndex(ui->cboxInterval->findText(QString::number(AppConfig::IntervalTcpServer)));
    connect(ui->cboxInterval, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));

    ui->cboxListenIP->setCurrentIndex(ui->cboxListenIP->findText(AppConfig::TcpListenIP));
    connect(ui->cboxListenIP, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));

    ui->txtListenPort->setText(QString::number(AppConfig::TcpListenPort));
    connect(ui->txtListenPort, SIGNAL(textChanged(QString)), this, SLOT(saveConfig()));

    ui->ckSelectAll->setChecked(AppConfig::SelectAllTcpServer);
    connect(ui->ckSelectAll, SIGNAL(stateChanged(int)), this, SLOT(saveConfig()));

    this->initTimer();
}

void frmTcpServer::saveConfig()
{
    AppConfig::HexSendTcpServer = ui->ckHexSend->isChecked();
    AppConfig::HexReceiveTcpServer = ui->ckHexReceive->isChecked();
    AppConfig::AsciiTcpServer = ui->ckAscii->isChecked();
    AppConfig::DebugTcpServer = ui->ckDebug->isChecked();
    AppConfig::AutoSendTcpServer = ui->ckAutoSend->isChecked();
    AppConfig::IntervalTcpServer = ui->cboxInterval->currentText().toInt();
    AppConfig::TcpListenIP = ui->cboxListenIP->currentText().trimmed();
    AppConfig::TcpListenPort = ui->txtListenPort->text().trimmed().toInt();
    AppConfig::SelectAllTcpServer = ui->ckSelectAll->isChecked();
    AppConfig::writeConfig();

    this->initTimer();
}

void frmTcpServer::initTimer()
{
    if (timer->interval() != AppConfig::IntervalTcpServer) {
        timer->setInterval(AppConfig::IntervalTcpServer);
    }

    if (AppConfig::AutoSendTcpServer) {
        if (!timer->isActive()) {
            timer->start();
        }
    } else {
        if (timer->isActive()) {
            timer->stop();
        }
    }
}

void frmTcpServer::append(int type, const QString &data, bool clear)
{
    static int currentCount = 0;
    static int maxCount = 100;
    QtHelper::appendMsg(ui->txtMain, type, data, maxCount, currentCount, clear, ui->ckShow->isChecked());
}

void frmTcpServer::connected(const QString &ip, int port)
{
    append(2, QString("[%1:%2] %3").arg(ip).arg(port).arg("客户端上线"));

    QString str = QString("%1:%2").arg(ip).arg(port);
    ui->listWidget->addItem(str);
    ui->labCount->setText(QString("共 %1 个客户端").arg(ui->listWidget->count()));
}

void frmTcpServer::disconnected(const QString &ip, int port)
{
    append(2, QString("[%1:%2] %3").arg(ip).arg(port).arg("客户端下线"));

    int row = -1;
    QString str = QString("%1:%2").arg(ip).arg(port);
    for (int i = 0; i < ui->listWidget->count(); i++) {
        if (ui->listWidget->item(i)->text() == str) {
            row = i;
            break;
        }
    }

    delete ui->listWidget->takeItem(row);
    ui->labCount->setText(QString("共 %1 个客户端").arg(ui->listWidget->count()));
}

void frmTcpServer::error(const QString &ip, int port, const QString &error)
{
    append(3, QString("[%1:%2] %3").arg(ip).arg(port).arg(error));
}

void frmTcpServer::sendData(const QString &ip, int port, const QString &data)
{
    append(0, QString("[%1:%2] %3").arg(ip).arg(port).arg(data));
}

void frmTcpServer::receiveData(const QString &ip, int port, const QString &data)
{
    append(1, QString("[%1:%2] %3").arg(ip).arg(port).arg(data));

    // 保存数据到数据库
    if (dbManager) {
        dbManager->saveData(ip, port, data);
    }

    // 自动回复"success"给发送数据的客户端
    if (isOk) {
        server->writeData(ip, port, "success");
    }
}

void frmTcpServer::on_btnListen_clicked()
{
    if (ui->btnListen->text() == "监听") {
        QString ip = ui->cboxListenIP->currentText().trimmed();
        int port = ui->txtListenPort->text().toInt();

        // 验证IP地址格式
        QRegularExpression ipRegex("^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$");
        if (!ipRegex.match(ip).hasMatch() && ip != "0.0.0.0") {
            append(3, "IP地址格式不正确");
            return;
        }

        // 验证端口范围
        if (port < 1 || port > 65535) {
            append(3, "端口号必须在1-65535之间");
            return;
        }

        isOk = server->listen(QHostAddress(ip), port);
        if (isOk) {
            append(2, QString("监听成功: %1:%2").arg(ip).arg(port));
            ui->btnListen->setText("关闭");
        } else {
            append(3, QString("监听失败: %1").arg(server->errorString()));
        }
    } else {
        isOk = false;
        server->stop();
        ui->btnListen->setText("监听");
    }
}

void frmTcpServer::on_btnSave_clicked()
{
    QString data = ui->txtMain->toPlainText();
    AppData::saveData(data);
    on_btnClear_clicked();
}

void frmTcpServer::on_btnClear_clicked()
{
    append(0, "", true);
}

void frmTcpServer::on_btnSend_clicked()
{
    if (!isOk) {
        return;
    }

    QString data = ui->cboxData->currentText();
    if (data.length() <= 0) {
        return;
    }

    if (ui->ckSelectAll->isChecked()) {
        server->writeData(data);
    } else {
        int row = ui->listWidget->currentRow();
        if (row >= 0) {
            QString str = ui->listWidget->item(row)->text();
            QStringList list = str.split(":");
            server->writeData(list.at(0), list.at(1).toInt(), data);
        }
    }
}

void frmTcpServer::on_btnRemove_clicked()
{
    if (ui->ckSelectAll->isChecked()) {
        server->remove();
    } else {
        int row = ui->listWidget->currentRow();
        if (row >= 0) {
            QString str = ui->listWidget->item(row)->text();
            QStringList list = str.split(":");
            server->remove(list.at(0), list.at(1).toInt());
        }
    }
}

void frmTcpServer::on_btnViewData_clicked()
{
    if (!dbManager) {
        append(3, "数据库未初始化!");
        return;
    }

    QList<QStringList> data = dbManager->getRecentData();

    QDialog dialog(this);
    dialog.setWindowTitle("查看历史数据");
    dialog.resize(1000, 600);

    QVBoxLayout *layout = new QVBoxLayout(&dialog);
    QTableWidget *table = new QTableWidget(&dialog);

    // 设置表格列数和标题
    table->setColumnCount(7);
    table->setHorizontalHeaderLabels(
        QStringList() << "IP地址" << "端口" << "时间戳" << "数值" << "类型" << "原始数据" << "记录时间");

    table->setRowCount(data.size());
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);

    // 填充数据
    for (int i = 0; i < data.size(); ++i) {
        const QStringList &record = data.at(i);
        for (int j = 0; j < record.size(); ++j) {
            QTableWidgetItem *item = new QTableWidgetItem(record.at(j));
            table->setItem(i, j, item);
        }
    }

    // 调整列宽
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    table->resizeColumnsToContents();
    table->horizontalHeader()->setStretchLastSection(true);

    layout->addWidget(table);
    dialog.exec();
}
