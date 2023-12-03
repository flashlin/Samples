Question: How to add a new B2B2C domain?
Question: How to add a new b2b2c domain?
Question: What is the process for adding a new b2b2c domain?
Question: What are the steps to add a new b2b2c domain?
Answer:
Assuming that NOC is ready to provide the following domain for us to add: abc.net

Here are the steps to add a new b2b2c domain:
1. Go to the http://dba-sb-prod.coreop.net/ Artimes website.
   Check whether these domains already exist in the InfoDb.AuthroizedDomain table.

2. Go to the https://forseti-api-a.sbotopex.com/swagger/index.html website.

3. If "abc.net" already exists in the InfoDb.AuthroizedDomain table:
   Perform a POST request to /api/public/AdminAuthorizedDomain/DeleteDomain to delete "abc.net" from the database.

4. Perform a POST request to /api/public/AdminAuthorizedDomain/InsertDomain to insert "abc.net" domain.

5. After the insertion is successful, check the result at http://forseti-api-a.sbotopex.com/domain.

6. If "abc.net" is intended for use in China, set the domain to "disable" in the database to prevent it from being used by agents from other countries.

7. If "abc.net" uses the HTTPS protocol, go to the http://dragonballz.coreop.net/ website and modify the AcceptHttpsDomain key value in the global_settings of the Forseti website to include "abc.net".

8. In the git Forseti pipeline, select "production:reload" to reload the global_settings configuration.

9. Restart the pods of the following projects:
   - Tera Backend
   - Pollux (SG Support or Hinoki should execute the necessary commands to restart pollux)

The following is the command to restart the pollux pod
```
gcloud auth list
gcloud auth login erwin.chang@appsafe.org
gcloud container clusters get-credentials host-prod --region asia-east1 --project host-prod-6751d597
kubectl rollout restart deployment tera-backend -n product-platform
kubectl rollout restart deployment pollux-asi -n airgile
kubectl rollout restart deployment pollux-bsi -n airgile
```

10. Go to https://atlas.coreop.net/reload and select odin -> "Reload Authorized Domain".

11. Use the new domain to explore our site and check if everything is working correctly.
    - Verify there are no issues when switching between casual site products.
    - Check desktop login and product switching.
    - Verify the register and deposit pages are functioning properly.
    - After binding the domain to an agent, check if the register page shows B2B2C register and the correct agent number.
    - After binding the domain to an agent, test typing the URL www.numbersite.com on a mobile device to see if it redirects to the casual site.

Question: How to convert a B2B2C domain to a B2C domain ?
Answer:
1. Go to the http://forseti-api-a.sbotopex.com website.
2. Select the "Demo CRUD" button.
3. As an example, search for the B2B2C domain "67146750.com" and disable it while providing the reason for the change.
   If there are any issues, ensure that frontline support is informed.

If you want to switch to the SBOBET domain:
Notify NOC to configure the domain as non-B2B2C.
Otherwise, the website will redirect to the casual site instead of the classic site (e.g., m.67146750.com).

In practice,
SBOBET should go to the classic site.
SBOTOP should redirect to the casual site (play.67146750.com).

Question: How to process VND B2B2C Manual Rebate for Insurance and Lucky?
Answer:
Here are the steps involved:
1. Prepare a CSV or Excel file from SMA containing the username and amount to credit.
2. Check if the file format is correct.
The file format should be like the following.
```
#,username,Refunds
1,1630198028,2000
2,1630197218,500
```
3.Please ask PO or DM to upload the file on Leo -> WMA ->"B2B2C Promo"
Done.

Question: How to check if a player has a deposit bonus?
Question: How can I determine if a player has a deposit bonus?
Question: How do I find out if a player has a deposit bonus?
Question: Is there a way to check if a player has a deposit bonus?
Question: How can I verify if a player has a deposit bonus?
Question: How can I confirm if a player has a deposit bonus?
Answer:
You can use the following tsql script to query.
```
select top 100  [UserName],
CASE PromotionCode
  WHEN 'TOPVN' THEN 'Sports Deposit Bonus'
  WHEN 'LCAS' then 'Casino Deposit Bonus'
END as PromotionType
from CustomerPromotions with (nolock)
where username in ('1630198028', '1620181772')
```

Question: How to enrollment Process for External Training?
Answer: 
Request External Training:
1. Engage in a discussion with DM and provide the following information.
2. Explain the reason behind your interest in joining this course.
3. Outline what you intend to bring back from the training.
4. Address any concerns related to SKA.
After External Training:
1. Complete the Department Training Report form within 3 days.
2. Submit the form to DM.
3. Schedule a review meeting with DM within 1 week.


Question: Why force the use of HTTPS on some domains?
Answer: The reason we do this is that our domains were once affected by advertising injection from internet service providers (ISPs) in other countries. 
By mandating the use of HTTPS, we can prevent ISPs from injecting ads into our domains.


Question: How to use TvpTable in Titansoft.DAL.Dapper and Titansoft.DAL.EF?
Question: How to use TvpTable in Titan?
Question: How to use TvpTable in Titansoft?
Answer: in Titansoft.DAL.Dapper
```C#
using Titansoft.DAL.Dapper;
using Titansoft.DAL.Dapper.Repositories;
...
   public async Task<List<GamesName>> GetTranslationByIds(List<int> ids)
   {
      return (await _client.QueryAsync<GamesName>("[dbo].[GetTranslation]", new
      {
         ids = GetTvpTable(ids),
         groupId = GamesNameGroupId,
      })).ToList();
   }
    
   private DataTable GetTvpTable(IEnumerable<int> idList)
   {
      var table = new DataTable();
      table.Columns.Add("value", typeof(int));
      foreach (var id in idList)
      {
         var row = table.NewRow();
         row["value"] = id;
         table.Rows.Add(row);
      }
      return table;
   }
```

in Titansoft.DAL.EF
```C#
using Titansoft.DAL.EF.Extensions.DB;
using Titansoft.DAL.EF.Extensions.DB.Models.Parameters;
using Titansoft.DAL.EF.Factory;
...
    public async Task UpsertSummaries(IEnumerable<TvpBetSummary> betSummaries)
    {
        var parameters = new List<TiSqlParameter>
        {
            new TiTableParameter
            {
                Name = "summaries",
                Type = "[dbo].[tvpBetSummary]",
                Value = betSummaries
            }
        };
        await _dbContext.ExecuteScalarAsync("[dbo].[UpsertSummaries]", parameters);
    }
```

Question: If you encounter a similar EF Exception problem as below.
```error
Titansoft.DAL.Dapper.Exceptions.DatabaseException: DatabaseClient Exception: System.InvalidOperationException: An enumerable sequence of parameters (arrays, lists, etc) is not allowed in this context at Dapper.SqlMapper.GetCacheInfo(Identity identity, Object exampleParameters, Boolean addToCache) in //Dapper/SqlMapper.cs:line 1706 at Dapper.SqlMapper.QueryAsync[T](IDbConnection cnn, Type effectiveType, CommandDefinition command) in //Dapper/SqlMapper.Async.cs:line 410 at Titansoft.DAL.Dapper.Repositories.DatabaseClient.<>c__DisplayClass20_0
The above error is due to using TiSqlParameter and TiTableParameter in Dapper.
```

or error
```error
Microsoft.Data.SqlClient.SqlException (PROMOTION-SCHEDULERS-8464998894-WSVDW:1)
Invalid column name 'Discriminator'.

Microsoft.Data.SqlClient.SqlException (0x80131904): Invalid column name 'Discriminator'.
   at Microsoft.Data.SqlClient.SqlConnection.OnError(SqlException exception, Boolean breakConnection, Action`1 wrapCloseInAction)
   at Microsoft.Data.SqlClient.SqlInternalConnection.OnError(SqlException exception, Boolean breakConnection, Action`1 wrapCloseInAction)
   at Microsoft.Data.SqlClient.TdsParser.ThrowExceptionAndWarning(TdsParserStateObject stateObj, Boolean callerHasConnectionLock, Boolean asyncClose)
   at Microsoft.Data.SqlClient.TdsParser.TryRun(RunBehavior runBehavior, SqlCommand cmdHandler, SqlDataReader dataStream, BulkCopySimpleResultSet bulkCopyHandler, TdsParserStateObject stateObj, Boolean& dataReady)
   at Microsoft.Data.SqlClient.SqlDataReader.TryConsumeMetaData()
   at Microsoft.Data.SqlClient.SqlDataReader.get_MetaData()
   at Microsoft.Data.SqlClient.SqlCommand.FinishExecuteReader(SqlDataReader ds, RunBehavior runBehavior, String resetOptionsString, Boolean isInternal, Boolean forDescribeParameterEncryption, Boolean shouldCacheForAlwaysEncrypted)
   at Microsoft.Data.SqlClient.SqlCommand.RunExecuteReaderTds(CommandBehavior cmdBehavior, RunBehavior runBehavior, Boolean returnStream, Boolean isAsync, Int32 timeout, Task& task, Boolean asyncWrite, Boolean inRetry, SqlDataReader ds, Boolean describeParameterEncryptionRequest)
   at Microsoft.Data.SqlClient.SqlCommand.RunExecuteReader(CommandBehavior cmdBehavior, RunBehavior runBehavior, Boolean returnStream, TaskCompletionSource`1 completion, Int32 timeout, Task& task, Boolean& usedCache, Boolean asyncWrite, Boolean inRetry, String method)
   at Microsoft.Data.SqlClient.SqlCommand.RunExecuteReader(CommandBehavior cmdBehavior, RunBehavior runBehavior, Boolean returnStream, String method)
   at Microsoft.Data.SqlClient.SqlCommand.ExecuteReader(CommandBehavior behavior)
   at Microsoft.Data.SqlClient.SqlCommand.ExecuteDbDataReader(CommandBehavior behavior)
   at Microsoft.EntityFrameworkCore.Storage.RelationalCommand.ExecuteReader(RelationalCommandParameterObject parameterObject)
   at Microsoft.EntityFrameworkCore.Query.Internal.SingleQueryingEnumerable`1.Enumerator.InitializeReader(Enumerator enumerator)
```

Answer:
```C#
public class CustomerVoucherRewardEntity : CustomerVoucherRewardBaseEntity
{
    public CustomerPromotionRewardEntity CustomerPromotionReward { get; set; }
}
```

Please check EntityConfiguration code, The following code snippet is incorrect. 
```C#
// use CustomerVoucherRewardEntity instead
public class CustomerVoucherRewardEntityConfiguration : IEntityTypeConfiguration<CustomerVoucherRewardBaseEntity>
{
    // use CustomerVoucherRewardEntity instead
    public void Configure(EntityTypeBuilder<CustomerVoucherRewardBaseEntity> builder)
    {
        builder.Build();
    }
}
```

The correct code should be as follows
```C#
public class CustomerVoucherRewardEntityConfiguration : IEntityTypeConfiguration<CustomerVoucherRewardEntity>
{
    public void Configure(EntityTypeBuilder<CustomerVoucherRewardEntity> builder)
    {
        builder.Build();
    }
}
```

Question: What are the things to be aware of when using a testing account?
Question: What are the precautions for using a testing account?
Answer: Recently, UA told me that they need to filtering testing account, 
but sometimes we create account not in test mode. 
This requires a lot of effort for them to filter those account.

I would like to set this rule, and there is no real user with this prefix now.
To create an account NOT in test mode for testing purpose, you prefix it with "rmdv"
ex: rmdvvnd001


Question: Why are all changes in the CI pipeline unable to execute? They remain in a pending state.
Answer: This information is current as of 2023-10-24. You can refer to this URL:
https://ironman.atlassian.net/wiki/spaces/ITR/pages/2833187369/Runner+spec


|Tag Name |CPU |Memory |Purpose
|--|--|--|--
|docker-small |1 |4 |build image
|build-small |1 |4 |
|kubectl |1 |4 |
|gcloud |1 |4 |
|dotnet-core |1 |4 
|docker-cli |1 |4
|docker-buildx |1 |4
|build-medium |2 |8 |build image
|build-large |4 |16 |
|docker-small |1 |4 |run docker image
|docker-medium-docker |2 |8
|docker-large |4 |16


Question: When you encounter the following error message on a CI pipeline, how do you resolve it?
```
Running ih giab-runner '5.3
Preparng he "kubernetes" execuor
Preparin envronmen
Natng ror pod hcm-runnerlrunner-duxjmxsyproec2t2concurrent8tmps o be running, status is Pending
ContainersNotInitiaLized: "containers with incompLete status: [init-permissions]"
ContainersNotReady: "containers with unready status: [buiLd helper]"

WARNING: Failed to pull image with policy: image pull failed: rpc error: code = Unknown desc = Error response from daemon:
Get "gttps://registery.gitlab.com/v2": proxyconnect tcp: dail tcp 10.1.1.2:328: connect: connection refused
```

Answer:
The proxy server is experiencing issues, possibly due to network problems, and cannot pull GitLab-related images. Please contact NOC for assistance


Question: How to access Redis Insight? Where can I find the website entrance?
Answer: Redis Insight has various entrances, much like different doors. These entrances are as follows:
* Staging Environment -> http://redis-insight-staging.coreop.net
* UAT Environment -> http://redis-insight-uat.coreop.net
* Production Environment -> http://redis-insight.coreop.net