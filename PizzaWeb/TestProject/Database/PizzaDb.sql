CREATE TABLE [dbo].[BannerTemplate]
(
    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [TemplateContent] [nvarchar](4000) NULL,
    [VariablesJson] [nvarchar](4000) NULL,
    [LastModifiedTime] [datetime] NOT NULL
    CONSTRAINT [UK_BannerTemplates] UNIQUE ([TemplateName] ASC)
)

CREATE TABLE [dbo].[Banner]
(
    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [OrderId] [int] NOT NULL DEFAULT(1),
    [BannerName] [varchar](50) NOT NULL,
    [VariableOptionsJson] [varchar](4000) NOT NULL,
    [LastModifiedTime] [datetime] NOT NULL DEFAULT (getdate()),
)

CREATE TABLE [dbo].[Resx]
(
    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
    [ISOLangCode] [varchar](30) NOT NULL,
    [VarType] [varchar](40) NOT NULL,
    [ResxName] [varchar](100) NOT NULL,
    [Content] [nvarchar](4000) NOT NULL,
    CONSTRAINT [UK_Resx] UNIQUE ([ISOLangCode] ASC, [ResxName] ASC, [VarType] ASC)
)

CREATE TABLE [dbo].[BannerShelf]
(
    [Uid] [Uniqueidentifier] PRIMARY KEY NOT NULL,
    [BannerName] [varchar](50) NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [TemplateContent] [nvarchar](4000) NOT NULL,
    [OrderId] [int] NOT NULL
)

CREATE TABLE [dbo].[VariableShelf]
(
   [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
   [Uid] [Uniqueidentifier] NOT NULL,
   [VarName] [varchar](50) NOT NULL,
   [ResxName] [varchar](100) NOT NULL,
   [ISOLangCode] [varchar](30) NOT NULL,
   [Content] [nvarchar](4000) NOT NULL
)

CREATE INDEX IX_VariableShelf
ON [VariableShelf](Uid,ISOLangCode,VarName,ResxName)
