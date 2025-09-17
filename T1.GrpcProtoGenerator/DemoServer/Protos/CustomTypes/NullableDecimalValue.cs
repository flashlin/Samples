namespace Roger.Protos.CustomTypes
{
    public partial class NullableDecimalValue
    {
        private const decimal NanoFactor = 1_000_000_000;

        public NullableDecimalValue(long units, int nanos, bool isNull)
        {
            Units = units;
            Nanos = nanos;
            IsNull = isNull;
        }

        public static implicit operator decimal?(NullableDecimalValue? grpcDecimal)
        {
            if (grpcDecimal == null || grpcDecimal.IsNull)
                return null;

            return grpcDecimal.Units + grpcDecimal.Nanos / NanoFactor;
        }

        public static implicit operator NullableDecimalValue?(decimal? value)
        {
            if (!value.HasValue)
            {
                return null;
            }

            var units = decimal.ToInt64(value.Value);
            var nanos = decimal.ToInt32((value.Value - units) * NanoFactor);
            return new NullableDecimalValue(units, nanos, false);
        }
    }
}