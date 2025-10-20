using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettledOrdersEntityConfiguration : IEntityTypeConfiguration<SettledOrdersEntity>
    {
        public void Configure(EntityTypeBuilder<SettledOrdersEntity> builder)
        {
            builder.ToTable("SettledOrders");


            builder.Property(x => x.TransactionId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
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

            builder.Property(x => x.CommissionableStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.TurnoverStake)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.WinLost)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.CommissionRate)
                .HasColumnType("decimal(5,4)")
                .IsRequired()
            ;

            builder.Property(x => x.Commission)
                .HasColumnType("decimal(19,6)")
                .IsRequired()
            ;

            builder.Property(x => x.PositionTaking)
                .HasColumnType("decimal(3,2)")
                .IsRequired()
            ;

            builder.Property(x => x.WinlostDate)
                .HasColumnType("date")
                .IsRequired()
            ;

            builder.Property(x => x.TransactionStatus)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IsResettled)
                .HasColumnType("tinyint(3,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Id)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

        }
    }
}
