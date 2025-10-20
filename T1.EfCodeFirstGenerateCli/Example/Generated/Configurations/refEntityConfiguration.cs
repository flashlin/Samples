using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class refEntityConfiguration : IEntityTypeConfiguration<refEntity>
    {
        public void Configure(EntityTypeBuilder<refEntity> builder)
        {
            builder.ToTable("ref");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.refno)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.rid)
                .HasColumnType("tinyint(3,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

        }
    }
}
