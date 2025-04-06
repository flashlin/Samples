using System.Threading.Tasks;
using T1.GrpcSourceGenerator;

namespace ConsoleTestApp
{
    public class GetUserReq
    {
        public int Id { get; set; }
        public bool IncludeDetails { get; set; }
    }

    /// <summary>
    /// 問候服務實現
    /// </summary>
    [GenerateGrpcService(typeof(IGreeter))]
    public class MyGreeter : IGreeter
    {
        /// <summary>
        /// 實現 SayHelloAsync 方法
        /// </summary>
        public Task<string> SayHelloAsync(string name)
        {
            return Task.FromResult($"Hello, {name}!");
        }
        
        /// <summary>
        /// 實現 GetUserInfoAsync 方法
        /// </summary>
        public Task<UserInfo> GetUserInfoAsync(GetUserReq req)
        {
            var userInfo = new UserInfo
            {
                Id = req.Id,
                Name = $"User-{req.Id}",
                Age = 30 + req.Id % 20,
                Details = req.IncludeDetails ? "This is the detailed information about the user." : null
            };
            
            return Task.FromResult(userInfo);
        }
        
        /// <summary>
        /// 實現 Add 方法
        /// </summary>
        public int Add(int a, int b)
        {
            return a + b;
        }
    }
} 