using System;

namespace Generated
{
    public class CountryPhoneCodeEntity
    {
        public int No { get; set; }
        public required string CountryName { get; set; }
        public required string PhoneCode { get; set; }
        public required string CountryCode { get; set; }
        public bool? Status { get; set; }
        public DateTime? ModifiedDate { get; set; }
        public int? ModifiedBy { get; set; }
        public DateTime TStamp { get; set; }
        public required string ContinentCode { get; set; }
        public required string ContinentName { get; set; }
        public byte? Continentorder { get; set; }
        public required string CurrencyDenied { get; set; }
        public bool? IsLayout { get; set; }
        public bool? IsDefault { get; set; }
    }
}
