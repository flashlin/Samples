-- init.sql
IF NOT EXISTS (
    SELECT name 
    FROM master.sys.databases 
    WHERE name = 'MotorDb'
)
BEGIN
    PRINT 'Creating MotorDb...'
    CREATE DATABASE MotorDb;
END
GO

USE MotorDb;

IF NOT EXISTS (
    SELECT TOP 1 * 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = 'Customer'
)
BEGIN
    CREATE TABLE [Customer] (
        [Id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        [Name] NVARCHAR(50) NOT NULL,
        [Email] VARCHAR(50) NOT NULL
    )
END
GO
