using System;

namespace Generated
{
    public class CountriesEntity
    {
        public int CountryID { get; set; }
        public required string CountryName { get; set; }
        public required string NativeName { get; set; }
        public required string ISO3166 { get; set; }
        public required string FIFACode { get; set; }
        public required string TelephoneCode { get; set; }
        public required string Remark { get; set; }
        public int Status { get; set; }
        public int CreatedBy { get; set; }
        public DateTime CreatedTime { get; set; }
        public int LastModifiedBy { get; set; }
        public DateTime LastModifiedTime { get; set; }
        public int Timezone { get; set; }
        public required string ContinentCode { get; set; }
        public required string ContinentName { get; set; }
        public byte? ContinentOrder { get; set; }
        public bool? isLeagueOnly { get; set; }
        public bool? isDefault { get; set; }
        public bool? isLayout { get; set; }
        public bool? IsEuropeanUnion { get; set; }
        public byte? RiskLevel { get; set; }
    }
}
