USE [PizzaDb]
GO

DECLARE @resxNames TVP_ResxNameVarType
INSERT @resxNames(ResxName, VarType)
VALUES ('SaltedChickenPizzaImage','Image(100,200)'),
('SaltedChickenPizzaTitle','String')

EXEC [dbo].[SP_GetResxNames] @resxNames

