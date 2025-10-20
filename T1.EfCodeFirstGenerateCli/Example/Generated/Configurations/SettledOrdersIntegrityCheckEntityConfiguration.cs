using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SettledOrdersIntegrityCheckEntityConfiguration : IEntityTypeConfiguration<SettledOrdersIntegrityCheckEntity>
    {
        public void Configure(EntityTypeBuilder<SettledOrdersIntegrityCheckEntity> builder)
        {
            builder.ToTable("SettledOrdersIntegrityCheck");


            builder.Property(x => x.LastCheckDateTime)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.LastCheckStatus)
                .HasColumnType("int(10,0)")
            ;

        }
    }
}
