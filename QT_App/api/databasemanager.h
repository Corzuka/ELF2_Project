// databasemanager.h
#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H

#include <QObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDateTime>

class DatabaseManager : public QObject
{
    Q_OBJECT
public:
    explicit DatabaseManager(QObject *parent = nullptr);
    ~DatabaseManager();

    bool initDatabase();
    bool saveData(const QString &ip, int port, const QString &data);
    QList<QStringList> getRecentData(int limit = 300);
    void cleanupOldData(int maxCount = 300);

private:
    QSqlDatabase db;
    QString dbPath;
};

#endif // DATABASEMANAGER_H
