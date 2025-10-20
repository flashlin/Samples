using System;

namespace Generated
{
    public class PhoneBetAccessEntity
    {
        public int ID { get; set; }
        public required string key { get; set; }
        public int account { get; set; }
        public DateTime expireDate { get; set; }
        public int status { get; set; }
        public int roleid { get; set; }
    }
}
