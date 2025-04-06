using System.Threading.Tasks;

namespace T1.GrpcSourceGenerator.Demo
{
    /// <summary>
    /// IGreeter 接口的實現類，使用 GenerateGrpcServiceAttribute 標記
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
        public Task<UserInfo> GetUserInfoAsync(int id, bool includeDetails)
        {
            var userInfo = new UserInfo
            {
                Id = id,
                Name = $"User-{id}",
                Age = 30 + id % 20,
                Details = includeDetails ? "This is the detailed information about the user." : null
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