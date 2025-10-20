using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class UnsettledOrdersEntityConfiguration : IEntityTypeConfiguration<UnsettledOrdersEntity>
    {
        public void Configure(EntityTypeBuilder<UnsettledOrdersEntity> builder)
        {
            builder.ToTable("UnsettledOrders");

            builder.HasKey(x => new { x.TransactionId, x.CustomerId });

            builder.Property(x => x.TransactionId)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransactionDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.PlayerStatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CurrencyId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RoleId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Stake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.ActualStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.CommissionRate)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
            ;

            builder.Property(x => x.PositionTaking)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

        }
    }
}
