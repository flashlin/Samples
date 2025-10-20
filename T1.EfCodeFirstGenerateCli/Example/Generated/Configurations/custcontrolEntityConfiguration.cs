using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class custcontrolEntityConfiguration : IEntityTypeConfiguration<custcontrolEntity>
    {
        public void Configure(EntityTypeBuilder<custcontrolEntity> builder)
        {
            builder.ToTable("custcontrol");

            builder.HasKey(x => x.rid);

            builder.Property(x => x.rid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.custid)
                .HasColumnType("int(10,0)")
                .IsRequired()
                .HasDefaultValue(1)
            ;

            builder.Property(x => x.leaguestr)
                .HasColumnType("varchar(8000)")
                .HasMaxLength(8000)
                .HasDefaultValue("1")
            ;

        }
    }
}
