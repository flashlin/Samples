-- init.sql
IF NOT EXISTS (
    SELECT name 
    FROM master.sys.databases 
    WHERE name = 'MotorDb'
)
BEGIN
    CREATE DATABASE MotorDb;
END

USE MotorDb;

IF NOT EXISTS (
    SELECT * 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = 'Customer'
)
BEGIN
    CREATE TABLE [dbo].[Customer] (
        [Id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        [Name] NVARCHAR(50) NOT NULL,
        [Email] NVARCHAR(50) NOT NULL
    )
END
