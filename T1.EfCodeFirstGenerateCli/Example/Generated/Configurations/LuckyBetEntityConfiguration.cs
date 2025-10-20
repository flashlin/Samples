using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class LuckyBetEntityConfiguration : IEntityTypeConfiguration<LuckyBetEntity>
    {
        public void Configure(EntityTypeBuilder<LuckyBetEntity> builder)
        {
            builder.ToTable("LuckyBet");


            builder.Property(x => x.CustId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IdleTime)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.ActiveTime)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.ButtonClickTime)
                .HasColumnType("datetime2")
            ;

            builder.Property(x => x.HostId)
                .HasColumnType("varchar(20)")
                .HasMaxLength(20)
            ;

        }
    }
}
