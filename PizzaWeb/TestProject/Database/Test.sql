USE [PizzaDb]
GO

DECLARE @resxNames TVP_ResxNameVarType
INSERT @resxNames(ResxName, VarType)
VALUES ('SaltedChickenPizzaImage','Image(100,200)'),
('SaltedChickenPizzaTitle','String')

EXEC [dbo].[SP_GetResxNames] @resxNames

    
INSERT StoreShelves(Title,Content,ImageName)   
VALUES
    ('Salted Chicken Pizza', 'Good', 'https://picsum.photos/300/200?random=5'),
    ('Sea Pizza', 'Good', 'https://picsum.photos/300/200?random=6'),
    ('Fish Pizza', 'Good', 'https://picsum.photos/300/200?random=7')
