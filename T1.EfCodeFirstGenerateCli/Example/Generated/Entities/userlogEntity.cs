using System;

namespace Generated
{
    public class userlogEntity
    {
        public int logid { get; set; }
        public required string userid { get; set; }
        public required string doing { get; set; }
        public DateTime? logdate { get; set; }
        public DateTime? tstamp { get; set; }
    }
}
