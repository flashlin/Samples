using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class MpCashUsedEntityConfiguration : IEntityTypeConfiguration<MpCashUsedEntity>
    {
        public void Configure(EntityTypeBuilder<MpCashUsedEntity> builder)
        {
            builder.ToTable("MpCashUsed");

            builder.HasKey(x => x.CustomerId);

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.recommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.mrecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.srecommend)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CashUsed)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.AgtCashUsed)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.MaCashUsed)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SmaCashUsed)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.ServiceProvider)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

            builder.Property(x => x.LastOrderOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.Username)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.IsOutstanding)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.tstamp)
                .HasColumnType("timestamp")
            ;

            builder.Property(x => x.SBStakeLimit)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.SBLimitExpiredDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.SBUsedStakeLimit)
                .HasColumnType("decimal(19,6)")
            ;

        }
    }
}
