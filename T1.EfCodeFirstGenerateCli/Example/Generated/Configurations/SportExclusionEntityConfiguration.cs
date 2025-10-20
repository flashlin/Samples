using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SportExclusionEntityConfiguration : IEntityTypeConfiguration<SportExclusionEntity>
    {
        public void Configure(EntityTypeBuilder<SportExclusionEntity> builder)
        {
            builder.ToTable("SportExclusion");

            builder.HasKey(x => new { x.CustID, x.SportID });

            builder.Property(x => x.CustID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.SportID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
