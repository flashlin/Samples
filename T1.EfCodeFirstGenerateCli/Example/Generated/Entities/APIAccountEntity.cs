using System;

namespace Generated
{
    public class APIAccountEntity
    {
        public int custid { get; set; }
        public required string username { get; set; }
        public required string company_name { get; set; }
        public required string ips { get; set; }
        public required string remark { get; set; }
        public int status { get; set; }
        public required string modifiedby { get; set; }
        public DateTime? modifieddate { get; set; }
    }
}
