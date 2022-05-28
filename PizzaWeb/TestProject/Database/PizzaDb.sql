CREATE TABLE [dbo].[BannerTemplates]
(
    [Id] [int] IDENTITY(1,1) NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [TemplateContent] [nvarchar](4000) NULL,
    [VariablesData] [nvarchar](4000) NULL,
    [LastModifiedTime] [datetime] NOT NULL,
    PRIMARY KEY CLUSTERED ([Id] ASC)
    CONSTRAINT [UK_BannerTemplates] UNIQUE NONCLUSTERED
    (
        [TemplateName] ASC
    )
)


CREATE TABLE [dbo].[Banners]
(
    [Id] [int] IDENTITY(1,1) NOT NULL,
    [TemplateName] [varchar](50) NOT NULL,
    [OrderId] [int] NOT NULL DEFAULT(1),
    [Name] [varchar](50) NOT NULL,
    [VariableOptions] [varchar](4000) NOT NULL,
    [LastModifiedTime] [datetime] NOT NULL DEFAULT (getdate()),
    PRIMARY KEY CLUSTERED ([Id] ASC)
)


CREATE TABLE [dbo].[Resx]
(
    [Id] [int] IDENTITY(1,1) NOT NULL,
    [ISOLangCode] [varchar](30) NOT NULL,
    [VarType] [varchar](40) NOT NULL,
    [Name] [varchar](100) NOT NULL,
    [Content] [nvarchar](4000) NOT NULL,
    PRIMARY KEY CLUSTERED([Id] ASC)
    CONSTRAINT [UK_Resx] UNIQUE NONCLUSTERED 
    ([ISOLangCode] ASC, [Name] ASC, [VarType] ASC
)