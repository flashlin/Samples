USE [master]
GO
CREATE DATABASE QueryDB
GO

USE [QueryDB]
GO

CREATE TABLE [SqlHistory]
(
   [ID] INT IDENTITY PRIMARY KEY,
   [SqlCode] NVARCHAR(2000) NOT NULL,
   [CreatedOn] DATETIME NOT NULL
)
GO

CREATE INDEX IDX_SqlHistory_CreatedOn
ON SqlHistory(CreatedOn)
GO

