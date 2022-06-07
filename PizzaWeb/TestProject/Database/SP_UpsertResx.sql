CREATE PROC SP_UpsertResx
    @resx TVP_Resx READONLY
AS BEGIN

    UPDATE r
    SET r.Content = d.Content
    FROM Resx as r
    JOIN @resx as d on r.ResxName=d.ResxName AND r.VarType=d.VarType AND r.IsoLangCode=d.IsoLangCode

    INSERT INTO Resx(ResxName, VarType, IsoLangCode, Content)
    SELECT d.ResxName, d.VarType, d.IsoLangCode, d.Content
    FROM @resx as d
    LEFT JOIN Resx as r on d.ResxName=r.ResxName AND d.VarType=r.VarType AND d.IsoLangCode=r.IsoLangCode
END    