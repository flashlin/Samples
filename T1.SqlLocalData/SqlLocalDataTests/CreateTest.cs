using System;
using System.IO;
using T1.SqlLocalData;
using Xunit;

namespace SqlLocalDataTests
{
    public class CreateTest : IDisposable
    {
        private string _instanceName = "localtest";
        private string _databaseFile = @"D:\Demo\test.mdf";
        private SqlLocalDb _localDb = new SqlLocalDb();

        public CreateTest()
        {
            CreateInstance();
        }

        private void CreateInstance()
        {
            if (!_localDb.IsInstanceExists(_instanceName))
            {
                return;
            }

            _localDb.CreateInstance(_instanceName);
            File.Delete(_databaseFile);
        }

        public void Dispose()
        {
            //_localDb.DeleteInstance();
        }

        [Fact]
        public void create_database()
        {
            _localDb.CreateDatabase(_instanceName, _databaseFile);
            Assert.True(File.Exists(_databaseFile));
        }
    }
}