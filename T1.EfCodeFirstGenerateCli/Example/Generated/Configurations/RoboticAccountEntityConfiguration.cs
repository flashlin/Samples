using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class RoboticAccountEntityConfiguration : IEntityTypeConfiguration<RoboticAccountEntity>
    {
        public void Configure(EntityTypeBuilder<RoboticAccountEntity> builder)
        {
            builder.ToTable("RoboticAccount");


            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.RoboticType)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.CreatedOn)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedOn)
                .HasColumnType("datetime")
            ;

        }
    }
}
