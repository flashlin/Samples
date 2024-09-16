using System.Data;
using System.Data.Common;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage;
using T1.EfCore;

public class DbCommandArgumentBuilder
{
    private readonly IRelationalTypeMappingSource _relationalTypeMappingSource;
    private readonly DbCommand _dbCommand;

    public DbCommandArgumentBuilder(DbContext dbContext, DbCommand dbCommand)
    {
        _dbCommand = dbCommand;
        _relationalTypeMappingSource = dbContext.GetService<IRelationalTypeMappingSource>();
    }
    
    public DbParameter CreateDbParameter(ConstantValue constantValue)
    {
        RelationalTypeMapping? relationalTypeMapping = null;

        if (constantValue.Property != null)
        {
            relationalTypeMapping = _relationalTypeMappingSource.FindMapping(constantValue.Property);
        }
        else if (constantValue.MemberInfo != null)
        {
            relationalTypeMapping = _relationalTypeMappingSource.FindMapping(constantValue.MemberInfo);
        }
        
        var dbParameterName = $"@p{constantValue.ArgumentIndex}";

        var dbParameter = relationalTypeMapping?.CreateParameter(_dbCommand, dbParameterName, constantValue.Value);
        if (dbParameter == null)
        {
            dbParameter = _dbCommand.CreateParameter();
            dbParameter.Direction = ParameterDirection.Input;
            dbParameter.Value = constantValue.Value;
            dbParameter.ParameterName = dbParameterName;
        }
        return dbParameter;
    }
}