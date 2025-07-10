// databasemanager.cpp
#include "databasemanager.h"
#include "qthelper.h"
#include <QSqlDatabase>
#include <QSqlError>
DatabaseManager::DatabaseManager(QObject *parent) : QObject(parent)
{
    dbPath = QtHelper::appPath() + "/data.db";
}

DatabaseManager::~DatabaseManager()
{
    if (db.isOpen()) {
        db.close();
    }
}

bool DatabaseManager::initDatabase() {
    // 检查 SQLite 驱动是否可用
    if (!QSqlDatabase::isDriverAvailable("QSQLITE")) {
        qDebug() << "Error: SQLite driver not loaded!";
        return false;
    }

    // 初始化数据库
    db = QSqlDatabase::addDatabase("QSQLITE", "TCP_SERVER_CONNECTION");
    db.setDatabaseName(dbPath);

    if (!db.open()) {
        qDebug() << "Database error:" << db.lastError().text();
        return false;
    }

    // 检查表是否存在，不存在则创建
    QSqlQuery query(db);
    if (!db.tables().contains("tcp_data")) {
        // 在initDatabase方法中修改表创建语句
        QString createTable = "CREATE TABLE tcp_data ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "ip TEXT NOT NULL, "
                              "port INTEGER NOT NULL, "
                              "timestamp TEXT, "      // 来自JSON的timestamp
                              "value TEXT, "         // 来自JSON的value
                              "type TEXT, "          // 来自JSON的type
                              "original_data TEXT, " // 原始完整数据
                              "db_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"; // 数据库记录时间
        if (!query.exec(createTable)) {
            qDebug() << "Failed to create table:" << query.lastError().text();
            return false;
        }
    }

    return true;
}

bool DatabaseManager::saveData(const QString &ip, int port, const QString &data)
{
    // 解析JSON数据
    QJsonParseError jsonError;
    QJsonDocument jsonDoc = QJsonDocument::fromJson(data.toUtf8(), &jsonError);

    QString timestamp, value, type;
    bool isJsonValid = false;

    if (jsonError.error == QJsonParseError::NoError && jsonDoc.isObject()) {
        QJsonObject jsonObj = jsonDoc.object();
        timestamp = jsonObj.value("timestamp").toString();
        value = jsonObj.value("value").toString();
        type = jsonObj.value("type").toString();
        isJsonValid = true;
    }

    QSqlQuery query(db);

    if (isJsonValid) {
        query.prepare("INSERT INTO tcp_data (ip, port, timestamp, value, type, original_data) "
                      "VALUES (:ip, :port, :timestamp, :value, :type, :original_data)");
        query.bindValue(":ip", ip);
        query.bindValue(":port", port);
        query.bindValue(":timestamp", timestamp);
        query.bindValue(":value", value);
        query.bindValue(":type", type);
        query.bindValue(":original_data", data);
    } else {
        // 如果不是有效的JSON，只保存原始数据
        query.prepare("INSERT INTO tcp_data (ip, port, original_data) "
                      "VALUES (:ip, :port, :original_data)");
        query.bindValue(":ip", ip);
        query.bindValue(":port", port);
        query.bindValue(":original_data", data);
    }

    if (!query.exec()) {
        qDebug() << "Failed to insert data:" << query.lastError().text();
        qDebug() << "Executed SQL:" << query.lastQuery();
        qDebug() << "Bound values:" << query.boundValues();
        return false;
    }

    cleanupOldData();
    return true;
}

QList<QStringList> DatabaseManager::getRecentData(int limit)
{
    QList<QStringList> result;
    QSqlQuery query(db);
    query.prepare("SELECT ip, port, timestamp, value, type, original_data, db_timestamp "
                  "FROM tcp_data ORDER BY db_timestamp DESC LIMIT ?");
    query.addBindValue(limit);

    if (query.exec()) {
        while (query.next()) {
            QStringList record;
            record << query.value(0).toString(); // ip
            record << query.value(1).toString(); // port
            record << query.value(2).toString(); // timestamp (from JSON)
            record << query.value(3).toString(); // value (from JSON)
            record << query.value(4).toString(); // type (from JSON)
            record << query.value(5).toString(); // original_data
            record << query.value(6).toString(); // db_timestamp
            result.append(record);
        }
    } else {
        qDebug() << "Failed to get recent data:" << query.lastError().text();
    }

    return result;
}



void DatabaseManager::cleanupOldData(int maxCount)
{
    QSqlQuery query(db);
    query.prepare("DELETE FROM tcp_data WHERE id NOT IN "
                  "(SELECT id FROM tcp_data ORDER BY timestamp DESC LIMIT ?)");
    query.addBindValue(maxCount);
    query.exec();
}

