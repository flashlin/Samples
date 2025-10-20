using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class JoinNowPromotionLogEntityConfiguration : IEntityTypeConfiguration<JoinNowPromotionLogEntity>
    {
        public void Configure(EntityTypeBuilder<JoinNowPromotionLogEntity> builder)
        {
            builder.ToTable("JoinNowPromotionLog");


            builder.Property(x => x.custId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.Turnover14)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.Winloss14)
                .HasColumnType("decimal(19,6)")
            ;

            builder.Property(x => x.date)
                .HasColumnType("datetime")
            ;

        }
    }
}
