CREATE TABLE [dbo].[BannerTemplates]
(
    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [TemplateContent] [nvarchar](4000) NULL,
    [VariablesData] [nvarchar](4000) NULL,
    [LastModifiedTime] [datetime] NOT NULL
    --CONSTRAINT [UK_BannerTemplates] UNIQUE ([TemplateName] ASC)
)


--CREATE TABLE [dbo].[Banners]
--(
--    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
--    [TemplateName] [varchar](50) NOT NULL,
--    [OrderId] [int] NOT NULL DEFAULT(1),
--    [Name] [varchar](50) NOT NULL,
--    [VariableOptions] [varchar](4000) NOT NULL,
--    [LastModifiedTime] [datetime] NOT NULL DEFAULT (getdate()),
--)


--CREATE TABLE [dbo].[Resx]
--(
--    [Id] [int] IDENTITY(1,1) PRIMARY KEY NOT NULL,
--    [ISOLangCode] [varchar](30) NOT NULL,
--    [VarType] [varchar](40) NOT NULL,
--    [Name] [varchar](100) NOT NULL,
--    [Content] [nvarchar](4000) NOT NULL,
--    CONSTRAINT [UK_Resx] UNIQUE ([ISOLangCode] ASC, [Name] ASC, [VarType] ASC)
--)