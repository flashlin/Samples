using System.Threading.Tasks;

namespace ConsoleTestApp
{
    /// <summary>
    /// 問候服務接口
    /// </summary>
    public interface IGreeter
    {
        /// <summary>
        /// 向指定的人問候
        /// </summary>
        /// <param name="name">姓名</param>
        /// <returns>問候語</returns>
        Task<string> SayHelloAsync(string name);
        
        /// <summary>
        /// 獲取用戶信息
        /// </summary>
        /// <param name="id">用戶 ID</param>
        /// <param name="includeDetails">是否包含詳細信息</param>
        /// <returns>用戶信息</returns>
        Task<UserInfo> GetUserInfoAsync(int id, bool includeDetails);
        
        /// <summary>
        /// 計算兩個數的和
        /// </summary>
        /// <param name="a">第一個數</param>
        /// <param name="b">第二個數</param>
        /// <returns>兩數之和</returns>
        int Add(int a, int b);
    }
    
    /// <summary>
    /// 用戶信息
    /// </summary>
    public class UserInfo
    {
        /// <summary>
        /// 用戶 ID
        /// </summary>
        public int Id { get; set; }
        
        /// <summary>
        /// 用戶名
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// 年齡
        /// </summary>
        public int Age { get; set; }
        
        /// <summary>
        /// 詳細信息
        /// </summary>
        public string Details { get; set; }
    }
} 