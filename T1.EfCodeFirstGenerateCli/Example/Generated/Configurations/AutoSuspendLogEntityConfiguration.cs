using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AutoSuspendLogEntityConfiguration : IEntityTypeConfiguration<AutoSuspendLogEntity>
    {
        public void Configure(EntityTypeBuilder<AutoSuspendLogEntity> builder)
        {
            builder.ToTable("AutoSuspendLog");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.AccountId)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Currency)
                .HasColumnType("char(3)")
                .HasMaxLength(3)
            ;

            builder.Property(x => x.MarketRate)
                .HasColumnType("decimal(12,8)")
            ;

            builder.Property(x => x.TotalBalanceInBaseCurrency)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.TotalOutstandingInBaseCurrency)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SuspendLimit)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SuspendedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.GroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.GroupName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SuspendedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
