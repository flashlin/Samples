using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class sma_bal_bbuEntityConfiguration : IEntityTypeConfiguration<sma_bal_bbuEntity>
    {
        public void Configure(EntityTypeBuilder<sma_bal_bbuEntity> builder)
        {
            builder.ToTable("sma_bal_bbu");


            builder.Property(x => x.cid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.amt)
                .HasColumnType("")
            ;

            builder.Property(x => x.status)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(0)
            ;

        }
    }
}
