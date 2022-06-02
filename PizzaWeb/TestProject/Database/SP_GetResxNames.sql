CREATE PROC SP_GetResxNames
    @resxNames TVP_ResxNameVarType READONLY
AS BEGIN
SELECT r.ResxName, r.VarType, r.IsoLangCode, r.Content
FROM Resx as r
         JOIN @resxNames as t on r.ResxName = t.ResxName and r.VarType = t.VarType
END
